import argparse

from kubernetes import client, config


def create_master(node, port, gpu):
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "master", "namespace": namespace},
        "spec": {
            "hostNetwork": True,
            "nodeSelector": {"kubernetes.io/hostname": node},
            "containers": [
                {
                    "name": "app",
                    "image": "asdfry/torch201-cuda118:accelerate-master",
                    "imagePullPolicy": "Always",
                    "command": [
                        "/bin/bash",
                        "-c",
                        f"/usr/sbin/sshd -p {port} && sleep infinity",
                    ],
                    "ports": [{"containerPort": port}],
                    "resources": {
                        "limits": {
                            gpu: "1",
                        }
                    },
                },
            ],
        },
    }
    v1.create_namespaced_pod(namespace, pod_manifest)


def create_worker(node, port, gpu):
    global worker_num
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": f"worker-{worker_num}", "namespace": namespace},
        "spec": {
            "hostNetwork": True,
            "nodeSelector": {"kubernetes.io/hostname": node},
            "containers": [
                {
                    "name": "app",
                    "image": "asdfry/torch201-cuda118:accelerate-worker",
                    "imagePullPolicy": "Always",
                    "command": [
                        "/bin/bash",
                        "-c",
                        f"/usr/sbin/sshd -p {port} && sleep infinity",
                    ],
                    "ports": [{"containerPort": port}],
                    "resources": {
                        "limits": {
                            gpu: "1",
                        }
                    },
                },
            ],
        },
    }
    v1.create_namespaced_pod(namespace, pod_manifest)
    worker_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--master_node_num", type=int, required=True)
    parser.add_argument("-s", "--slot_size", type=int, required=True)
    parser.add_argument("-t", "--total_node", type=int, required=True)
    parser.add_argument("-gm", "--gpu_master", type=str, required=True)
    parser.add_argument("-gw", "--gpu_worker", type=str, required=True)
    args = parser.parse_args()

    config.load_kube_config()
    v1 = client.CoreV1Api()

    namespace = "jsh"
    worker_num = 1

    gpu = "ten1010.io/gpu-nvidia-a100-pcie-40gb"
    node = f"k8s-node-{args.master_node_num}"
    create_master(node, 1041, gpu)
    for i in range(1, args.slot_size):
        create_worker(node, 1041 + i, gpu)

    gpu = "ten1010.io/gpu-tesla-t4"
    for i in range(1, args.total_node):
        node = f"k8s-node-{args.master_node_num + i}"
        for i in range(0, args.slot_size):
            create_worker(node, 1041 + i, gpu)
