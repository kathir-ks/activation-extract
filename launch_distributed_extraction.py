"""
Python launcher for distributed activation extraction

More flexible than bash script - supports various deployment methods:
1. GCP TPU VMs (using gcloud)
2. SSH to remote machines
3. Local testing (simulates distributed by running sequentially)

Usage:
    # Launch on GCP TPU VMs
    python launch_distributed_extraction.py --deployment gcp_tpu --tpu_names tpu-vm-{0..31} --zone us-central1-a

    # Launch via SSH
    python launch_distributed_extraction.py --deployment ssh --hosts machine-{0..31}.example.com

    # Local testing (sequential)
    python launch_distributed_extraction.py --deployment local --total_machines 4
"""

import argparse
import subprocess
import time
from pathlib import Path
import json


class DistributedLauncher:
    """Launch and monitor distributed extraction jobs"""

    def __init__(self, deployment: str, total_machines: int, gcs_bucket: str,
                 model_path: str = "KathirKs/qwen-2.5-0.5b",
                 dataset_name: str = "HuggingFaceFW/fineweb-edu",
                 dataset_config: str = "sample-10BT",
                 dataset_split: str = "train",
                 batch_size: int = 8,
                 max_seq_length: int = 2048,
                 layers_to_extract: str = "10 11 12 13 14 15 16 17 18 19 20 21 22 23",
                 shard_size_gb: float = 1.0,
                 max_samples: int = None,
                 working_dir: str = "/home/kathirks_gc/torch_xla/qwen",
                 logs_dir: str = "./logs"):

        self.deployment = deployment
        self.total_machines = total_machines
        self.gcs_bucket = gcs_bucket
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.layers_to_extract = layers_to_extract
        self.shard_size_gb = shard_size_gb
        self.max_samples = max_samples
        self.working_dir = working_dir

        # Create logs directory
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.processes = []

    def build_command(self, machine_id: int) -> str:
        """Build the extraction command for a specific machine"""
        cmd = f"""python extract_activations_fineweb.py \
--machine_id {machine_id} \
--total_machines {self.total_machines} \
--model_path {self.model_path} \
--dataset_name {self.dataset_name} \
--dataset_config {self.dataset_config} \
--dataset_split {self.dataset_split} \
--batch_size {self.batch_size} \
--max_seq_length {self.max_seq_length} \
--layers_to_extract {self.layers_to_extract} \
--shard_size_gb {self.shard_size_gb} \
--upload_to_gcs \
--gcs_bucket {self.gcs_bucket} \
--compress_shards \
--delete_local_after_upload \
--verbose"""

        if self.max_samples:
            cmd += f" --max_samples {self.max_samples}"

        return cmd

    def launch_gcp_tpu(self, tpu_names: list, zone: str):
        """Launch on GCP TPU VMs"""
        print("="*70)
        print(f"LAUNCHING ON GCP TPU VMs")
        print(f"Zone: {zone}")
        print(f"Machines: {len(tpu_names)}")
        print("="*70)

        for machine_id, tpu_name in enumerate(tpu_names[:self.total_machines]):
            cmd = self.build_command(machine_id)
            log_file = self.logs_dir / f"machine_{machine_id}.log"

            # Build gcloud command
            gcloud_cmd = [
                "gcloud", "compute", "tpus", "tpu-vm", "ssh", tpu_name,
                "--zone", zone,
                "--command", f"cd {self.working_dir} && {cmd}"
            ]

            print(f"  Launching machine {machine_id} ({tpu_name})...")

            # Launch in background
            with open(log_file, 'w') as f:
                proc = subprocess.Popen(
                    gcloud_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
                self.processes.append({
                    'machine_id': machine_id,
                    'tpu_name': tpu_name,
                    'process': proc,
                    'log_file': log_file
                })

            time.sleep(2)  # Small delay

        print(f"\n✓ All {len(self.processes)} jobs launched!")

    def launch_ssh(self, hosts: list):
        """Launch via SSH"""
        print("="*70)
        print(f"LAUNCHING VIA SSH")
        print(f"Machines: {len(hosts)}")
        print("="*70)

        for machine_id, host in enumerate(hosts[:self.total_machines]):
            cmd = self.build_command(machine_id)
            log_file = self.logs_dir / f"machine_{machine_id}.log"

            # Build SSH command
            ssh_cmd = [
                "ssh", host,
                f"cd {self.working_dir} && {cmd}"
            ]

            print(f"  Launching machine {machine_id} ({host})...")

            # Launch in background
            with open(log_file, 'w') as f:
                proc = subprocess.Popen(
                    ssh_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
                self.processes.append({
                    'machine_id': machine_id,
                    'host': host,
                    'process': proc,
                    'log_file': log_file
                })

            time.sleep(2)

        print(f"\n✓ All {len(self.processes)} jobs launched!")

    def launch_local(self):
        """Launch locally (sequential, for testing)"""
        print("="*70)
        print(f"LAUNCHING LOCALLY (SEQUENTIAL)")
        print(f"Simulating {self.total_machines} machines")
        print("="*70)

        for machine_id in range(self.total_machines):
            cmd = self.build_command(machine_id)
            log_file = self.logs_dir / f"machine_{machine_id}.log"

            print(f"  Running machine {machine_id}...")

            # Run sequentially
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    f"cd {self.working_dir} && {cmd}",
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )

                if result.returncode != 0:
                    print(f"    ✗ Machine {machine_id} failed (see {log_file})")
                else:
                    print(f"    ✓ Machine {machine_id} completed")

    def monitor(self):
        """Monitor running processes"""
        if not self.processes:
            print("No background processes to monitor (local mode runs sequentially)")
            return

        print("\n" + "="*70)
        print("MONITORING JOBS")
        print("="*70)

        while True:
            running = []
            completed = []
            failed = []

            for job in self.processes:
                status = job['process'].poll()
                if status is None:
                    running.append(job['machine_id'])
                elif status == 0:
                    completed.append(job['machine_id'])
                else:
                    failed.append(job['machine_id'])

            print(f"\r  Running: {len(running)} | Completed: {len(completed)} | Failed: {len(failed)}", end="")

            if not running:
                print("\n")
                break

            time.sleep(10)

        print("\n" + "="*70)
        print("ALL JOBS FINISHED")
        print("="*70)
        print(f"  Completed: {len(completed)}")
        print(f"  Failed: {len(failed)}")

        if failed:
            print(f"\n  Failed machines: {failed}")
            print(f"  Check logs in: {self.logs_dir}/")

    def print_info(self):
        """Print launch information"""
        print("\n" + "="*70)
        print("CONFIGURATION")
        print("="*70)
        print(f"  Deployment: {self.deployment}")
        print(f"  Total machines: {self.total_machines}")
        print(f"  GCS bucket: gs://{self.gcs_bucket}/activations_fineweb/")
        print(f"  Dataset: {self.dataset_name} ({self.dataset_config})")
        print(f"  Model: {self.model_path}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Max sequence length: {self.max_seq_length}")
        print(f"  Layers: {self.layers_to_extract}")
        print(f"  Shard size: {self.shard_size_gb} GB")
        if self.max_samples:
            print(f"  Max samples per machine: {self.max_samples}")
        print(f"  Logs: {self.logs_dir}/")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Launch distributed activation extraction")

    # Deployment method
    parser.add_argument('--deployment', type=str, required=True,
                       choices=['gcp_tpu', 'ssh', 'local'],
                       help="Deployment method")

    # Common args
    parser.add_argument('--total_machines', type=int, default=32,
                       help="Total number of machines")
    parser.add_argument('--gcs_bucket', type=str, required=True,
                       help="GCS bucket name")

    # GCP TPU specific
    parser.add_argument('--tpu_names', type=str, nargs='+',
                       help="TPU VM names (for gcp_tpu deployment)")
    parser.add_argument('--zone', type=str,
                       help="GCP zone (for gcp_tpu deployment)")

    # SSH specific
    parser.add_argument('--hosts', type=str, nargs='+',
                       help="Host names/IPs (for ssh deployment)")

    # Model and dataset
    parser.add_argument('--model_path', type=str, default="KathirKs/qwen-2.5-0.5b")
    parser.add_argument('--dataset_name', type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument('--dataset_config', type=str, default="sample-10BT")
    parser.add_argument('--dataset_split', type=str, default="train")

    # Extraction config
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--layers_to_extract', type=str,
                       default="10 11 12 13 14 15 16 17 18 19 20 21 22 23")
    parser.add_argument('--shard_size_gb', type=float, default=1.0)
    parser.add_argument('--max_samples', type=int, help="Max samples per machine (for testing)")

    # Paths
    parser.add_argument('--working_dir', type=str, default="/home/kathirks_gc/torch_xla/qwen")
    parser.add_argument('--logs_dir', type=str, default="./logs")

    # Monitor
    parser.add_argument('--monitor', action='store_true',
                       help="Monitor jobs after launching (only for gcp_tpu/ssh)")

    args = parser.parse_args()

    # Validate deployment-specific args
    if args.deployment == 'gcp_tpu':
        if not args.tpu_names or not args.zone:
            parser.error("--tpu_names and --zone required for gcp_tpu deployment")
    elif args.deployment == 'ssh':
        if not args.hosts:
            parser.error("--hosts required for ssh deployment")

    # Create launcher
    launcher = DistributedLauncher(
        deployment=args.deployment,
        total_machines=args.total_machines,
        gcs_bucket=args.gcs_bucket,
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        layers_to_extract=args.layers_to_extract,
        shard_size_gb=args.shard_size_gb,
        max_samples=args.max_samples,
        working_dir=args.working_dir,
        logs_dir=args.logs_dir
    )

    launcher.print_info()

    # Launch based on deployment method
    if args.deployment == 'gcp_tpu':
        launcher.launch_gcp_tpu(args.tpu_names, args.zone)
    elif args.deployment == 'ssh':
        launcher.launch_ssh(args.hosts)
    elif args.deployment == 'local':
        launcher.launch_local()

    # Monitor if requested
    if args.monitor and args.deployment != 'local':
        launcher.monitor()

    print("\nMonitor progress:")
    print(f"  Logs: tail -f {args.logs_dir}/machine_*.log")
    print(f"  GCS: gsutil ls gs://{args.gcs_bucket}/activations_fineweb/")


if __name__ == '__main__':
    main()