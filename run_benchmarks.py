import subprocess

# 要执行的两个 Python 脚本路径
scripts = ["bench_normal.py", "bench_financial.py"]

for script in scripts:
    print(f"\n🚀 正在运行：{script}")
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 运行失败：{script}\n错误信息：{e}")
