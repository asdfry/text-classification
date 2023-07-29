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
                    "image": "asdfry/torch201-cuda118:master",
                    "imagePullPolicy": "Always",
                    "command": [
                        "/bin/bash",
                        "-c",
                        f"sed -i 's/^#Port 22$/Port {port}/' /etc/ssh/sshd_config && /usr/sbin/sshd && sleep infinity",
                    ],
                    "volumeMounts": [
                        {"name": "data", "mountPath": "/root/data"},
                        {"name": "logs", "mountPath": "/root/logs"},
                        {"name": "pem", "mountPath": "/root/.ssh/aws-ten-jsh.pem"},
                    ],
                    "ports": [{"containerPort": port}],
                    "resources": {
                        "limits": {
                            gpu: "1",
                        }
                    },
                },
            ],
            "volumes": [
                {
                    "name": "data",
                    "hostPath": {
                        "path": "/home/jsh/workspaces/projects/text-classification/data",
                        "type": "Directory",
                    },
                },
                {
                    "name": "logs",
                    "hostPath": {
                        "path": "/home/jsh/workspaces/projects/text-classification/logs",
                        "type": "Directory",
                    },
                },
                {
                    "name": "pem",
                    "hostPath": {
                        "path": "/home/jsh/.ssh/aws-ten-jsh.pem",
                        "type": "File",
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
                    "image": "asdfry/torch201-cuda118:worker",
                    "imagePullPolicy": "Always",
                    "command": [
                        "/bin/bash",
                        "-c",
                        f"sed -i 's/^#Port 22$/Port {port}/' /etc/ssh/sshd_config && /usr/sbin/sshd && sleep infinity",
                    ],
                    "volumeMounts": [
                        {"name": "data", "mountPath": "/root/data"},
                    ],
                    "ports": [{"containerPort": port}],
                    "resources": {
                        "limits": {
                            gpu: "1",
                        }
                    },
                },
            ],
            "volumes": [
                {
                    "name": "data",
                    "hostPath": {
                        "path": "/home/jsh/workspaces/projects/text-classification/data",
                        "type": "Directory",
                    },
                },
            ],
        },
    }
    v1.create_namespaced_pod(namespace, pod_manifest)
    worker_num += 1


if __name__ == "__main__":
    config.load_kube_config()
    v1 = client.CoreV1Api()

    namespace = "jsh"
    worker_num = 1
    node_num = 2
    slot_count = 2
    total_node = 2

    gpu = "ten1010.io/gpu-nvidia-a100-pcie-40gb"
    node = f"k8s-node-{node_num}"
    create_master(node, 1041, gpu)
    for i in range(1, slot_count):
        create_worker(node, 1041 + i, gpu)

    gpu = "ten1010.io/gpu-tesla-t4"
    for i in range(1, total_node):
        node = f"k8s-node-{node_num + i}"
        for i in range(0, slot_count):
            create_worker(node, 1041 + i, gpu)
