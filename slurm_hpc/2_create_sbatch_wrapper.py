import sys
import subprocess
subprocess.run(["sh","2_create_sbatch_jobs.sh",sys.argv[1], sys.argv[2]])