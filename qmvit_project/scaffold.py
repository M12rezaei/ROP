"""
Updated scaffold that uses the provided Swin-UNet and QMViT implementations.
This script orchestrates segmentation training, QMViT training, and ML pipelines
and is meant as a project entry point for the zone-free dataset.
"""
import argparse, os, subprocess, sys, shutil
from pathlib import Path

# --- Helper function to get the base environment with updated PYTHONPATH ---
def get_env_with_pythonpath():
    """Returns a copy of the current environment with the project root added to PYTHONPATH."""
    # Find the root of the project (the directory containing 'scaffold.py')
    project_root = str(Path(__file__).parent.resolve())
    
    # Copy current environment variables
    env = os.environ.copy()
    
    # Prepend the project root to the PYTHONPATH
    current_pythonpath = env.get('PYTHONPATH', '')
    if current_pythonpath:
        # Use os.pathsep (';' on Windows) to separate paths
        env['PYTHONPATH'] = f"{project_root}{os.pathsep}{current_pythonpath}"
    else:
        env['PYTHONPATH'] = project_root
        
    return env


# --- Functions to run training scripts ---

def run_segmentation_train(data_dir, out_dir, epochs=3):
    script = Path(__file__).parent / 'scripts' / 'segmentation_train.py'
    cmd = [sys.executable, str(script), 
           '--data_dir', str(Path(data_dir)/'segmentation'), 
           '--output_dir', str(out_dir+'/segmentation'), 
           '--epochs', str(epochs)]
    
    # Use the environment with the updated PYTHONPATH
    env = get_env_with_pythonpath()
    
    print("Running segmentation training:", " ".join(cmd))
    # FIX APPLIED: pass the updated environment to the subprocess
    return subprocess.call(cmd, env=env)


def run_qmvit_train(data_dir, out_dir, task='stages', epochs=3):
    script = Path(__file__).parent / 'scripts' / 'qmvit_train.py'
    csv_path = Path(data_dir)/'train.csv'
    cmd = [sys.executable, str(script), 
           '--data_dir', str(data_dir), 
           '--output_dir', str(out_dir+'/qmvit_'+task), 
           '--task', task, 
           '--epochs', str(epochs)]
    if csv_path.exists():
        cmd += ['--csv', str(csv_path)]
    
    # Use the environment with the updated PYTHONPATH
    env = get_env_with_pythonpath()

    print("Running QMViT training:", " ".join(cmd))
    # FIX APPLIED: pass the updated environment to the subprocess
    return subprocess.call(cmd, env=env)


# --- Main execution block ---

def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.do_seg:
        rc = run_segmentation_train(args.data_dir, str(out), epochs=args.epochs)
        if rc != 0:
            print("Warning: segmentation training returned non-zero exit code", rc)
    if args.do_qmvit:
        rc = run_qmvit_train(args.data_dir, str(out), task=args.task, epochs=args.epochs)
        if rc != 0:
            print("Warning: qmvit training returned non-zero exit code", rc)
    print("Completed run. Check output directory:", out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help='Root data dir (contains stages/ plus/ and segmentation/ optionally)')
    p.add_argument('--output_dir', default='./outputs', help='Root output dir')
    p.add_argument('--do_seg', action='store_true', help='Run segmentation training')
    p.add_argument('--do_qmvit', action='store_true', help='Run QMViT classification training')
    p.add_argument('--task', default='stages', choices=['stages', 'plus'], help='Classification task for QMViT (stages or plus)')
    p.add_argument('--epochs', type=int, default=5, help='Number of epochs to run')
    args = p.parse_args()
    main(args)