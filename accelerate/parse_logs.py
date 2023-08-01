import re

parsed_list = []

for i in [1, 2, 4]:
    temp_dict = {"gpu": i}
    with open(f"./logs/log_gpu_{i}.log", "r") as f:
        lines = f.readlines()
        for line in lines:
            sh = re.search(r"epoch (\d)] elapsed time: (.+) sec", line)
            if sh:
                epoch = int(sh.group(1))
                elapsed = float(sh.group(2))
                temp_dict[f"epoch_{epoch}"] = elapsed
        parsed_list.append(temp_dict)

print(f"{'Elapsed time per GPU':-^130}")
for parsed in parsed_list:
    temp_str = ""
    for idx, value in enumerate(parsed.values()):
        if idx == 0:
            temp_str += f"[{value} GPU] "
        else:
            value = round(value, 6)
            temp_str += f"{idx} Epoch: {str(value): >10}\t"
    print(temp_str)

print(f"{'Training speed per GPU':-^130}")
for gpu, parsed in enumerate(parsed_list):
    temp_str = ""
    for idx, value in enumerate(parsed.values()):
        if idx == 0:
            temp_str += f"[{value} GPU] "
        else:
            value = round(parsed_list[0][f"epoch_{idx}"] / (value * 2**gpu), 3)
            temp_str += f"{idx} Epoch: {str(value): >10}\t"
    print(temp_str)
