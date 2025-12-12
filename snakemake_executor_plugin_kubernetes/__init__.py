import base64
from dataclasses import dataclass, field
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Self
import uuid

import kubernetes
import kubernetes.config
import kubernetes.client

from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.remote import RemoteExecutor
from snakemake_interface_executor_plugins.settings import (
    ExecutorSettingsBase,
    CommonSettings,
)
from snakemake_interface_executor_plugins.jobs import (
    JobExecutorInterface,
)
from snakemake_interface_common.exceptions import WorkflowError
from snakemake_interface_executor_plugins.settings import DeploymentMethod


@dataclass
class PersistentVolume:
    name: str
    path: Path

    @classmethod
    def parse(cls, arg: str) -> Self:
        spec = arg.split(":")
        if len(spec) != 2:
            raise WorkflowError(
                f"Invalid persistent volume spec ({arg}), has to be <name>:<path>."
            )
        name, path = spec
        return cls(name=name, path=Path(path))

    def unparse(self) -> str:
        return f"{self.name}:{self.path}"


def parse_persistent_volumes(args: List[str]) -> List[PersistentVolume]:
    return [PersistentVolume.parse(arg) for arg in args]


def unparse_persistent_volumes(args: List[PersistentVolume]) -> List[str]:
    return [arg.unparse() for arg in args]


def parse_bool(arg: str) -> bool:
    # Snakemake CLI parser for booleans is bugged.
    #   - booleans that default to "True" don't get namespaced with the plugin
    #     name like other options, which produces a crash. (e.g.
    #     `--kubernetes-no-foo` becomes `--no-foo` and therefore never makes it
    #     to this plugin).
    #   - optional boolean values are ignored, and only the presence or absence
    #     of the option is taken into account. (e.g. `--kubernetes-privileged
    #     true` is the same as `--kubernetes-privileged false` and will set the
    #     privileged flag). This only way to not enable the privileged flag is
    #     to not pass it to snakemake. Bruh.
    # Therefore, we implement this custom parser that just works.
    return arg in ["1", "true"]


def unparse_bool(arg: bool) -> str:
    return bool(arg).lower()


def parse_annotations(args: List[str]) -> Dict[str, str]:
    annotations = {}
    for arg in args:
        if "=" not in arg:
            raise WorkflowError(
                f"Invalid annotation spec ({arg}), has to be <key>=<value>."
            )
        key, value = arg.split("=", 1)
        annotations[key] = value
    return annotations


def unparse_annotations(annotations: Dict[str, str]) -> List[str]:
    return [f"{key}={value}" for key, value in annotations.items()]


@dataclass
class ExecutorSettings(ExecutorSettingsBase):
    namespace: str = field(
        default="default", metadata={"help": "The namespace to use for submitted jobs."}
    )
    cpu_scalar: float = field(
        default=0.95,
        metadata={
            "help": "K8s reserves some proportion of available CPUs for its own use. "
            "So, where an underlying node may have 8 CPUs, only e.g. 7600 milliCPUs "
            "are allocatable to k8s pods (i.e. snakemake jobs). As 8 > 7.6, k8s can't "
            "find a node with enough CPU resource to run such jobs. This argument acts "
            "as a global scalar on each job's CPU request, so that e.g. a job whose "
            "rule definition asks for 8 CPUs will request 7600m CPUs from k8s, "
            "allowing it to utilise one entire node. N.B: the job itself would still "
            "see the original value, i.e. as the value substituted in {threads}."
        },
    )
    service_account_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "This argument allows the use of customer service "
            "accounts for "
            "kubernetes pods. If specified, serviceAccountName will "
            "be added to the "
            "pod specs. This is e.g. needed when using workload "
            "identity which is enforced "
            "when using Google Cloud GKE Autopilot."
        },
    )
    privileged: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Create privileged containers for jobs.",
            "parse_func": parse_bool,
            "unparse_func": unparse_bool,
        },
    )
    persistent_volumes: List[PersistentVolume] = field(
        default_factory=list,
        metadata={
            "help": "Mount the given persistent volumes under the given paths in each "
            "job container (<name>:<path>). ",
            "parse_func": parse_persistent_volumes,
            "unparse_func": unparse_persistent_volumes,
            "nargs": "+",
        },
    )
    omit_job_cleanup: bool = field(
        default=False,
        metadata={
            "help": "Do not delete jobs after they have finished or failed. "
            "This is useful for debugging, or if your k8s cluster performs "
            "automatic cleanups."
        },
    )
    nvidia_runtime_class_name: Optional[str] = field(
        default=None,
        metadata={"help": "Runtime class name to use for NVIDIA jobs."},
    )
    gpu_overcommit: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Allow GPU overcommitment.",
            "parse_func": parse_bool,
            "unparse_func": unparse_bool,
        },
    )
    scheduler_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Set the scheduler name for the pod spec. "
            "This allows using custom schedulers (e.g., volcano, kube-batch). "
            "If specified, schedulerName will be added to the pod spec."
        },
    )
    annotations: Dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": "Add custom annotations to the pod template metadata. "
            "Format: <key>=<value>. Can be specified multiple times.",
            "parse_func": parse_annotations,
            "unparse_func": unparse_annotations,
            "nargs": "+",
        },
    )


# Required:
# Specify common settings shared by various executors.
common_settings = CommonSettings(
    # define whether your executor plugin executes locally
    # or remotely. In virtually all cases, it will be remote execution
    # (cluster, cloud, etc.). Only Snakemake's standard execution
    # plugins (snakemake-executor-plugin-dryrun, snakemake-executor-plugin-local)
    # are expected to specify False here.
    non_local_exec=True,
    # Define whether your executor plugin implies that there is no shared
    # filesystem (True) or not (False).
    # This is e.g. the case for cloud execution.
    implies_no_shared_fs=True,
    job_deploy_sources=True,
    pass_default_storage_provider_args=True,
    pass_default_resources_args=True,
    pass_envvar_declarations_to_cmd=False,
    auto_deploy_default_storage_provider=True,
)


# Required:
# Implementation of your executor
class Executor(RemoteExecutor):
    def __post_init__(self):
        # Attempt loading kube_config or in-cluster config
        try:
            kubernetes.config.load_kube_config()
        except kubernetes.config.config_exception.ConfigException:
            kubernetes.config.load_incluster_config()

        self.k8s_cpu_scalar = self.workflow.executor_settings.cpu_scalar
        self.k8s_service_account_name = (
            self.workflow.executor_settings.service_account_name
        )
        self.kubeapi = kubernetes.client.CoreV1Api()
        self.batchapi = kubernetes.client.BatchV1Api()
        self.namespace = self.workflow.executor_settings.namespace
        self.secret_files = {}
        self.run_namespace = str(uuid.uuid4())
        self.secret_envvars = {}
        self.register_secret()
        self.log_path = self.workflow.persistence.aux_path / "kubernetes-logs"
        self.log_path.mkdir(exist_ok=True, parents=True)
        self.container_image = self.workflow.remote_execution_settings.container_image
        self.privileged = self.workflow.executor_settings.privileged
        self.persistent_volumes = self.workflow.executor_settings.persistent_volumes
        self.nvidia_runtime_class_name = (
            self.workflow.executor_settings.nvidia_runtime_class_name
        )
        self.gpu_overcommit = self.workflow.executor_settings.gpu_overcommit
        self.scheduler_name = self.workflow.executor_settings.scheduler_name
        self.annotations = self.workflow.executor_settings.annotations

        self.logger.info(f"Using {self.container_image} for Kubernetes jobs.")

    def run_job(self, job: JobExecutorInterface):
        # Implement here how to run a job.
        # You can access the job's resources, etc.
        # via the job object.
        # After submitting the job, you have to call
        # self.report_job_submission(job_info).
        # with job_info being of type
        # snakemake_interface_executor_plugins.executors.base.SubmittedJobInfo.
        # Convert job.resources to a normal dict first (fix for membership checks).
        resources_dict = dict(job.resources)

        exec_job = self.format_job_exec(job)
        self.logger.debug(f"Executing job: {exec_job}")

        # Kubernetes silently does not submit a job if the name is too long
        # therefore, we ensure that it is not longer than snakejob+uuid.
        jobid = "snakejob-{}".format(
            get_uuid(f"{self.run_namespace}-{job.jobid}-{job.attempt}")
        )

        body = kubernetes.client.V1Job()
        body.metadata = kubernetes.client.V1ObjectMeta(labels={"app": "snakemake"})
        body.metadata.name = jobid

        # Container setup
        container = kubernetes.client.V1Container(name="snakemake")
        container.image = self.container_image
        container.command = shlex.split("/bin/sh")
        container.args = ["-c", exec_job]
        container.working_dir = "/workdir"
        container.volume_mounts = [
            kubernetes.client.V1VolumeMount(name="workdir", mount_path="/workdir"),
        ]

        # Volume mounts
        for pvc in self.persistent_volumes:
            container.volume_mounts.append(
                kubernetes.client.V1VolumeMount(name=pvc.name, mount_path=str(pvc.path))
            )

        # Node selector
        node_selector = {}
        if "machine_type" in resources_dict.keys():
            node_selector["node.kubernetes.io/instance-type"] = resources_dict[
                "machine_type"
            ]
            self.logger.debug(f"Set node selector for machine type: {node_selector}")

        # Initialize PodSpec
        pod_spec = kubernetes.client.V1PodSpec(
            containers=[container], node_selector=node_selector, restart_policy="Never"
        )

        template_metadata = None
        if self.annotations:
            template_metadata = kubernetes.client.V1ObjectMeta(annotations=self.annotations)
            self.logger.debug(f"Set template annotations: {self.annotations}")

        body.spec = kubernetes.client.V1JobSpec(
            backoff_limit=0,
            template=kubernetes.client.V1PodTemplateSpec(
                metadata=template_metadata,
                spec=pod_spec,
            ),
        )

        # Add toleration for GPU nodes if GPU is requested
        if "gpu" in resources_dict:
            # Manufacturer logic
            manufacturer = resources_dict.get("gpu_manufacturer", None)
            if not manufacturer:
                raise WorkflowError(
                    "GPU requested but no manufacturer set. "
                    "Use gpu_manufacturer='nvidia' or 'amd'."
                )
            manufacturer_lc = manufacturer.lower()
            if manufacturer_lc == "nvidia":
                # Toleration for nvidia.com/gpu
                if pod_spec.tolerations is None:
                    pod_spec.tolerations = []
                pod_spec.tolerations.append(
                    kubernetes.client.V1Toleration(
                        key="nvidia.com/gpu",
                        operator="Equal",
                        value="present",
                        effect="NoSchedule",
                    )
                )
                self.logger.debug(
                    f"Added toleration for NVIDIA GPU: {pod_spec.tolerations}"
                )

            elif manufacturer_lc == "amd":
                # Toleration for amd.com/gpu
                if pod_spec.tolerations is None:
                    pod_spec.tolerations = []
                pod_spec.tolerations.append(
                    kubernetes.client.V1Toleration(
                        key="amd.com/gpu",
                        operator="Equal",
                        value="present",
                        effect="NoSchedule",
                    )
                )
                self.logger.debug(
                    f"Added toleration for AMD GPU: {pod_spec.tolerations}"
                )

            else:
                raise WorkflowError(
                    f"Unsupported GPU manufacturer '{manufacturer}'. "
                    "Must be 'nvidia' or 'amd'."
                )

        # Add service account name if provided
        if self.k8s_service_account_name:
            pod_spec.service_account_name = self.k8s_service_account_name
            self.logger.debug(
                f"Set service account name: {self.k8s_service_account_name}"
            )

        # Add scheduler name if provided
        if self.scheduler_name:
            pod_spec.scheduler_name = self.scheduler_name
            self.logger.debug(f"Set scheduler name: {self.scheduler_name}")

        # Workdir volume
        workdir_volume = kubernetes.client.V1Volume(name="workdir")
        workdir_volume.empty_dir = kubernetes.client.V1EmptyDirVolumeSource()
        pod_spec.volumes = [workdir_volume]

        for pvc in self.persistent_volumes:
            volume = kubernetes.client.V1Volume(name=pvc.name)
            volume.persistent_volume_claim = (
                kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=pvc.name
                )
            )
            pod_spec.volumes.append(volume)

        # Env vars
        container.env = []
        for key, e in self.secret_envvars.items():
            envvar = kubernetes.client.V1EnvVar(name=e)
            envvar.value_from = kubernetes.client.V1EnvVarSource()
            envvar.value_from.secret_key_ref = kubernetes.client.V1SecretKeySelector(
                key=key, name=self.run_namespace
            )
            container.env.append(envvar)

        # Request resources
        self.logger.debug(f"Job resources: {resources_dict}")
        container.resources = kubernetes.client.V1ResourceRequirements()
        container.resources.requests = {}

        scale_value = resources_dict.get("scale", 1)

        container.resources.limits = {}

        # CPU and memory requests
        cores = resources_dict.get("_cores", 1)
        container.resources.requests["cpu"] = "{}m".format(
            int(cores * self.k8s_cpu_scalar * 1000)
        )

        if not scale_value:
            container.resources.limits["cpu"] = "{}m".format(int(cores * 1000))

        if "mem_mb" in resources_dict:
            mem_mb = resources_dict["mem_mb"]
            container.resources.requests["memory"] = "{}M".format(mem_mb)
            container.resources.limits["memory"] = "{}M".format(mem_mb)
        # Disk
        if "disk_mb" in resources_dict:
            disk_mb = int(resources_dict.get("disk_mb", 1024))
            container.resources.requests["ephemeral-storage"] = f"{disk_mb}M"
            if not scale_value:
                container.resources.limits["ephemeral-storage"] = f"{disk_mb}M"

        # Request GPU resources if specified
        if "gpu" in resources_dict:
            gpu_count = str(resources_dict["gpu"])
            # For nvidia, K8s expects nvidia.com/gpu; for amd, we use amd.com/gpu.
            # But let's keep nvidia.com/gpu
            # for both if the cluster doesn't differentiate.
            # If your AMD plugin uses a different name, update accordingly:
            manufacturer = resources_dict.get("gpu_manufacturer", "").lower()
            identifier = {
                "nvidia": "nvidia.com/gpu",
                "amd": "amd.com/gpu",
            }.get(manufacturer, "nvidia.com/gpu")
            container.resources.requests[identifier] = gpu_count
            if not scale_value or not self.gpu_overcommit:
                container.resources.limits[identifier] = gpu_count
            if identifier == "nvidia.com/gpu":
                if self.nvidia_runtime_class_name is not None:
                    pod_spec.runtime_class_name = self.nvidia_runtime_class_name
                if (
                    DeploymentMethod.APPTAINER
                    in self.workflow.deployment_settings.deployment_method
                ):
                    envvar = kubernetes.client.V1EnvVar(name="APPTAINER_NV")
                    envvar.value = "true"
                    container.env.append(envvar)

        # Privileged mode
        if self.privileged or (
            DeploymentMethod.APPTAINER
            in self.workflow.deployment_settings.deployment_method
        ):
            container.security_context = kubernetes.client.V1SecurityContext(
                privileged=True
            )
            self.logger.debug("Container set to run in privileged mode.")

        self.logger.debug(f"k8s pod resources: {container.resources}")

        # Assign the modified container back to the spec
        body.spec.containers = [container]

        # Serialize and log the pod specification
        import json

        self.logger.debug("Pod specification:")
        self.logger.debug(json.dumps(body.to_dict(), indent=2))

        # Try creating the pod with exception handling
        try:
            pod = self._kubernetes_retry(
                lambda: self.batchapi.create_namespaced_job(self.namespace, body)
            )
        except kubernetes.client.rest.ApiException as e:
            self.logger.error(f"Failed to create pod: {e}")
            raise WorkflowError(f"Failed to create pod: {e}")

        self.logger.info(f"Get status with: kubectl describe job {jobid}")

        self.report_job_submission(
            SubmittedJobInfo(job=job, external_jobid=jobid, aux={"pod": pod})
        )

    async def check_active_jobs(
        self, active_jobs: List[SubmittedJobInfo]
    ) -> AsyncGenerator[SubmittedJobInfo, None]:
        # Check the status of active jobs.

        # You have to iterate over the given list active_jobs.
        # For jobs that have finished successfully, you have to call
        # self.report_job_success(job).
        # For jobs that have errored, you have to call
        # self.report_job_error(job).
        # Jobs that are still running have to be yielded.
        #
        # For queries to the remote middleware, please use
        # self.status_rate_limiter like this:
        #
        # async with self.status_rate_limiter:
        #    # query remote middleware here
        self.logger.debug(f"Checking status of {len(active_jobs)} jobs")
        for j in active_jobs:
            async with self.status_rate_limiter:
                try:
                    res = self._kubernetes_retry(
                        lambda j=j: self.batchapi.read_namespaced_job_status(
                            j.external_jobid, self.namespace
                        )
                    )
                except kubernetes.client.rest.ApiException as e:
                    self.logger.error(f"ApiException when checking pod status: {e}")
                    continue
                except WorkflowError as e:
                    self.logger.error(f"WorkflowError when checking pod status: {e}")
                    continue

                if res is None:
                    msg = (
                        "Unknown job {jobid}. Has the job been deleted manually?"
                    ).format(jobid=j.external_jobid)
                    self.logger.error(msg)
                    self.report_job_error(j, msg=msg)
                    continue

                # Sometimes, just checking the status of a job is not enough, because
                # apparently, depending on the cluster setup, there can be additional
                # containers injected into pods that will prevent the job to detect
                # that a pod is already terminated.
                # We therefore check the status of the snakemake container in addition
                # to the job status.
                pods = self._kubernetes_retry(
                    lambda j=j: self.kubeapi.list_namespaced_pod(
                        namespace=self.namespace,
                        label_selector=f"job-name={j.external_jobid}",
                    )
                )
                assert len(pods.items) <= 1
                if pods.items and pods.items[0].status.container_statuses is not None:
                    pod = pods.items[0]
                    snakemake_container = [
                        container
                        for container in pod.status.container_statuses
                        if container.name == "snakemake"
                    ][0]
                    snakemake_container_exit_code = (
                        snakemake_container.state.terminated.exit_code
                        if snakemake_container.state.terminated is not None
                        else None
                    )
                    pod_name = pod.metadata.name
                else:
                    snakemake_container = None
                    snakemake_container_exit_code = None
                    pod_name = None

                if (res.status.failed and res.status.failed > 0) or (
                    snakemake_container_exit_code is not None
                    and snakemake_container_exit_code != 0
                ):
                    if pod_name is not None:
                        assert snakemake_container is not None
                        kube_log = self.log_path / f"{j.external_jobid}.log"
                        with open(kube_log, "w") as f:

                            def read_log(
                                pod_name=pod_name,
                                container_name=snakemake_container.name,
                            ):
                                return self.kubeapi.read_namespaced_pod_log(
                                    name=pod_name,
                                    namespace=self.namespace,
                                    container=container_name,
                                    previous=True,
                                )

                            kube_log_content = self._kubernetes_retry(read_log)
                            print(kube_log_content, file=f)
                        aux_logs = [str(kube_log)]
                        msg = ""
                    else:
                        msg = (
                            " For details, please issue:\n"
                            f"kubectl describe job {j.external_jobid}. "
                            "Further, make sure to clean up the failed job "
                            "manually in case it is not deleted automatically: "
                            "kubectl delete job {j.external_jobid}."
                        )
                        aux_logs = []

                    self.logger.error(f"Job {j.external_jobid} failed.{msg}")
                    self.report_job_error(j, msg=msg, aux_logs=aux_logs)

                    if (
                        pod_name is not None
                        and not self.workflow.executor_settings.omit_job_cleanup
                    ):
                        self._kubernetes_retry(
                            lambda j=j: self.safe_delete_job(
                                j.external_jobid, ignore_not_found=True
                            )
                        )
                elif (res.status.succeeded and res.status.succeeded >= 1) or (
                    snakemake_container_exit_code == 0
                ):
                    # finished
                    self.logger.info(f"Job {j.external_jobid} succeeded.")
                    self.report_job_success(j)

                    if not self.workflow.executor_settings.omit_job_cleanup:
                        self._kubernetes_retry(
                            lambda j=j: self.safe_delete_job(
                                j.external_jobid, ignore_not_found=True
                            )
                        )
                else:
                    # still active
                    self.logger.debug(f"Job {j.external_jobid} is still active.")
                    yield j

    def cancel_jobs(self, active_jobs: List[SubmittedJobInfo]):
        # Cancel all active jobs.
        for j in active_jobs:
            self._kubernetes_retry(
                lambda jobid=j.external_jobid: self.safe_delete_job(
                    jobid, ignore_not_found=True
                )
            )

    def shutdown(self):
        self.unregister_secret()
        super().shutdown()

    def register_secret(self):
        import kubernetes.client

        secret = kubernetes.client.V1Secret()
        secret.metadata = kubernetes.client.V1ObjectMeta()
        # create a random uuid
        secret.metadata.name = self.run_namespace
        secret.type = "Opaque"
        secret.data = {}

        for name, value in self.envvars().items():
            key = name.lower()
            secret.data[key] = base64.b64encode(value.encode()).decode()
            self.secret_envvars[key] = name

        # Test if the total size of the configMap exceeds 1MB
        config_map_size = sum(
            [len(base64.b64decode(v)) for k, v in secret.data.items()]
        )
        if config_map_size > 1048576:
            raise WorkflowError(
                "The total size of the included files and other Kubernetes secrets "
                f"is {config_map_size}, exceeding the 1MB limit.\n"
            )

        self.kubeapi.create_namespaced_secret(self.namespace, secret)

    def unregister_secret(self):
        import kubernetes.client

        self._kubernetes_retry(
            lambda: self.kubeapi.delete_namespaced_secret(
                self.run_namespace,
                self.namespace,
                body=kubernetes.client.V1DeleteOptions(),
            )
        )

    def safe_delete_job(self, jobid, ignore_not_found=True):
        import kubernetes.client

        body = kubernetes.client.V1DeleteOptions(propagation_policy="Foreground")
        self.logger.debug(f"Deleting job {jobid} in namespace {self.namespace}")
        try:
            self.batchapi.delete_namespaced_job(
                jobid, self.namespace, body=body
            )
        except kubernetes.client.rest.ApiException as e:
            if e.status == 404 and ignore_not_found:
                self.logger.debug(
                    "[WARNING] 404 not found when trying to delete the job: {jobid}\n"
                    "[WARNING] Ignore this error\n".format(jobid=jobid)
                )
            else:
                raise e

    def _reauthenticate_and_retry(self, func=None):
        import kubernetes

        # Unauthorized.
        # Reload config in order to ensure token is
        # refreshed. Then try again.
        self.logger.info("Trying to reauthenticate")
        kubernetes.config.load_kube_config()
        subprocess.run(["kubectl", "get", "nodes"])

        self.kubeapi = kubernetes.client.CoreV1Api()
        self.batchapi = kubernetes.client.BatchV1Api()

        try:
            self.register_secret()
        except kubernetes.client.rest.ApiException as e:
            if e.status == 409 and e.reason == "Conflict":
                self.logger.warning(
                    "409 conflict ApiException when registering secrets"
                )
                self.logger.warning(e)
            else:
                raise WorkflowError(
                    e,
                    "This is likely a bug in "
                    "https://github.com/kubernetes-client/python.",
                )

        if func:
            return func()

    def _kubernetes_retry(self, func) -> Any:
        import kubernetes
        import urllib3

        with self.lock:
            try:
                return func()
            except kubernetes.client.rest.ApiException as e:
                if e.status == 401:
                    # Unauthorized.
                    # Reload config in order to ensure token is
                    # refreshed. Then try again.
                    return self._reauthenticate_and_retry(func)
                raise WorkflowError("Kubernetes request failed.", e)
            # Handling timeout that may occur in case of GKE master upgrade
            except urllib3.exceptions.MaxRetryError:
                self.logger.warning(
                    "Request time out! "
                    "check your connection to Kubernetes master"
                    "Workflow will pause for 5 minutes to allow any update "
                    "operations to complete"
                )
                time.sleep(300)
                try:
                    return func()
                except Exception as e:
                    # Still can't reach the server after 5 minutes
                    raise WorkflowError(
                        e,
                        "Error 111 connection timeout, please check"
                        " that the k8s cluster master is reachable!",
                    )


UUID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/snakemake/snakemake-executor-plugin-kubernetes",
)


def get_uuid(name):
    return uuid.uuid5(UUID_NAMESPACE, name)
