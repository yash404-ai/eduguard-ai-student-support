import sys, importlib
sys.path.insert(0, __file__.rsplit("/scripts", 1)[0])
names = sys.argv[1:] or ["uci", "xapi", "student_perf", "oulad"]
for n in names:
    importlib.import_module(f"eduguard_ml.train_{n}").main()
