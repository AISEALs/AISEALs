search_words=$1
change_file=$2

real_file="test.txt"
#real_file="new_label.txt"

old_label="生活经历"
change_label="生活压力"

echo "wanted $old_label -> $change_label:"
grep "$search_words" $real_file

if [ "$change_file" != "" ]; then
	sed -i -E "/$1/s/$old_label/$change_label/" $real_file
	echo $?
	echo "changed line $old_label -> $change_label"
	grep "$search_words" $real_file
fi
