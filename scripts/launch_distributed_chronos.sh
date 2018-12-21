#!/bin/bash
function usage()
{
    echo "Usage: $0 [-n <NUM_NODES>][-p <NUM_PROCESSES_PER_NODE>][-f <COMMAND_FILE> | <COMMAND>]"
}

while getopts n:p:f: parm ; do
case $parm in
  n)
    nnodes=$OPTARG
    ;;
  p)
    nproc_per_node=$OPTARG
    ;;
  f)
    cmd_file=$OPTARG
    ;;
  *)
    usage
    echo "Invalid argument"
esac
done

shift $((OPTIND-1))

if  [[ -z $cmd_file ]]; then
  training_script=$*
else
  if [[ ! -f "$cmd_file" ]]; then
    usage
    echo "File not found : $cmd_file"
    exit 1
  fi
  training_script=$(cat "$cmd_file")
fi

hostgroup=fblearner_ash_bigbasin_fair
rndvpath="/mnt/vol/gfsai-east/ai-group/users/${USER}/chronos/rendezvous/"
solibdir="/mnt/vol/gfsai-east/ai-group/users/${USER}"
cpu_per_proc=6
cpu_per_node=$((nproc_per_node*cpu_per_proc))

if [ "$nnodes" -eq 1 ] && [ "$nproc_per_node" -eq 1 ]
then
  jobid=$(echo "AF_PATH=${solibdir} ${training_script}" \
      | /usr/local/chronos/scripts/crun --hostgroup "${hostgroup}" --gpu "${nproc_per_node}" --cpu "${cpu_per_node}")
elif [ "$nnodes" -eq 1 ]
then
  jobid=$(echo "AF_PATH=${solibdir} /usr/local/fbcode/gcc-5-glibc-2.23/bin/mpirun -n ${nproc_per_node} ${training_script} --enable_distributed" \
      | /usr/local/chronos/scripts/crun --hostgroup "${hostgroup}" --gpu "${nproc_per_node}" --cpu "${cpu_per_node}")
else
  rand=$(echo -n $RANDOM | sha256sum | cut -c1-9)
  echo "MultiNode training for wav2letter++ on AML cluster needs some adhoc modifications to the code. Please talk to @vineelkpratap."
  exit 1
  jobid=$(echo "AF_PATH=${solibdir} /usr/local/fbcode/gcc-5-glibc-2.23/bin/mpirun -n ${nproc_per_node} ${training_script} --enable_distributed --rndv_filepath=${rndvpath}/\${CHRONOS_JOB_ID}_${rand}" \
      | /usr/local/chronos/scripts/crun --hostgroup "${hostgroup}" --gang-size "${nnodes}" --gang-rack-affinity --gpu "${nproc_per_node}" --cpu "${cpu_per_node}")
fi

echo "Job ID: ${jobid}"
echo "  https://our.intern.facebook.com/intern/bunny/?x+${jobid}"
