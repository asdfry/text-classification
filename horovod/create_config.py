def write_master_config(network_addr: str, host_addr: int, port: int):
    hostname = f"{network_addr}.{host_addr}"
    with open(".ssh/config", "a") as f:
        f.write(f"Host master\n")
        f.write(f"    HostName {hostname}\n")
        f.write(f"    Port {port}\n")
        f.write(f"    User {user}\n")
        f.write(f"    IdentityFile {identity_file}\n")
        f.write(f"    StrictHostKeyChecking no\n\n")
    with open("run.sh", "a") as f:
        f.write(f"horovodrun -np $1 -H master:1")


def write_worker_config(network_addr: str, host_addr: int, port: int):
    global worker_num
    hostname = f"{network_addr}.{host_addr}"
    with open(".ssh/config", "a") as f:
        f.write(f"Host worker-{worker_num}\n")
        f.write(f"    HostName {hostname}\n")
        f.write(f"    Port {port}\n")
        f.write(f"    User {user}\n")
        f.write(f"    IdentityFile {identity_file}\n")
        f.write(f"    StrictHostKeyChecking no\n\n")
    with open("run.sh", "a") as f:
        f.write(f",worker-{worker_num}:1")
    worker_num += 1

def write_last_line():
    with open("run.sh", "a") as f:
        f.write(f" python3 train_multi.py -e 1 -b 4 -g 1 -hf\n")


if __name__ == "__main__":
    worker_num = 1
    user = input("Input User: ")
    identity_file = input("Input Identity File: ")
    slot_count = int(input("Input Slot Count: "))

    network_addr = input("Input Master's Network Address: ")
    host_addr = int(input("Input Master's Host Address: "))

    write_master_config(network_addr, host_addr, 1041)
    for i in range(1, slot_count):
        write_worker_config(network_addr, host_addr + i, 1041 + i)

    while True:
        host_addr = int(input("Input Worker's Host Address: "))
        if host_addr == -1:
            write_last_line()
            print("Exit")
            break
        for i in range(0, slot_count):
            write_worker_config(network_addr, host_addr + i, 1041 + i)
