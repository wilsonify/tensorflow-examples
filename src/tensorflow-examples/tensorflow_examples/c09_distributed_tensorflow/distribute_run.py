import subprocess


def main():
    subprocess.Popen('python distribute.py --job_name="ps" --task_index=0', shell=True)
    subprocess.Popen('python distribute.py --job_name="worker" --task_index=0', shell=True)
    subprocess.Popen('python distribute.py --job_name="worker" --task_index=1', shell=True)
    subprocess.Popen('python distribute.py --job_name="worker" --task_index=2', shell=True)


if __name__ == "__main__":
    main()
