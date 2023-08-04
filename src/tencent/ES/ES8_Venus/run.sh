#/user/bin

echo "nargs:$#, args:$*"

usage() { echo "Usage: $0 model_version lr sigma [-s START_DATA_HOUR:YYYY-MM-DD/HH] [-r rank_type]" 1>&2; exit 1; }
cd ES8_Venus

if [ $# -lt 3 ]; then
  usage
fi
export MODEL_VERSION=$1
shift
export PART=$1
shift
export LR=$1
shift
export SIGMA=$1
shift

while getopts ":s:r:n:" o; do
    case "${o}" in
        s)
            s=${OPTARG}
            ;;
        r)
            rank_type=${OPTARG}
            ;;
        n)
            is_norm=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
echo $((OPTIND-1))
shift $((OPTIND-1))

if [ -z "${s}" ]; then
  echo "s is empty str, use last hour. please explicit send [-s %YYYY-MM-DD%/%HH%] params:", ${s}
  s=$(date -d "1 hour ago" +"%Y-%m-%d/%H")
fi

if [ -z "${is_norm}" ]
  then
    export ISNORM=1
  else
    export ISNORM=${is_norm}
fi

export START_DATA_HOUR=$s

sh run_hourly_new.sh
ret=$?
echo "run_hourly.sh, ret = $ret"
if [ $ret -ne 0 ];then
  exit 1
fi

