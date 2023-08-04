import argparse
import os
import pprint
import traceback
import pandas as pd
import pkuseg

from data_processor.processor_manager import get_processor


class Item:
    def __init__(self, id, label_by_machine, label_by_machine_prob, label_by_people, people, content):
        self.id = id
        self.label_by_machine = label_by_machine
        self.label_by_machine_prob = label_by_machine_prob
        self.label_by_people = label_by_people
        self.people = people
        self.content = content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--is_debug",
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help="use debug mode")
    parser.add_argument(
        "--base_path",
        type=str,
        default="/Users/jiananliu/work/AISEALs/data/text_classification",
        help="refer to data_processor path tree")
    parser.add_argument(
        "--task_name",
        type=str,
        default="tribe_labels",
        help="task name")
    parser.add_argument(
        "--task_id",
        type=str,
        default="20190529",
        help="task id")
    parser.add_argument(
        "--label_name",
        type=str,
        default="心情_情绪_想法表达",
        help="refer to data_processor path tree")

    FLAGS, unparsed = parser.parse_known_args()
    pp = pprint.PrettyPrinter().pprint
    pp("FLAGS: " + str(FLAGS))
    pp("unparsed: " + str(unparsed))
    print(FLAGS.is_debug)

    print("base_path:" + str(FLAGS.base_path))

    # labels path: FLAGS.base_path + FLAGS.task_name + FLAGS.task_id
    processor = get_processor(FLAGS.base_path, FLAGS.task_name, FLAGS.task_id)

    label_classes = processor.get_labels()

    file_name = "/Users/jiananliu/Downloads/total.txt"

    user_dict = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data/my_dict.txt"
    )
    seg = pkuseg.pkuseg(user_dict=user_dict)

    items = []
    with open(file_name, encoding='utf8', errors='ignore') as f:
        for line in f:
            # print(line)
            try:
                sp = line.strip('\n').split("\t")
                labels = sp[-3].strip(",")
                label_list = labels.split(",")
                label_list = list(filter(lambda x: x in label_classes, label_list))
                if len(label_list) ==0 or (len(label_list) == 1 and label_list[0] == '个人自拍'):
                    continue
                for label in label_list:
                    content = sp[-1]
                    content = " ".join(seg.cut(content.replace(" ", "")))
                    items.append(Item(sp[0], sp[1], sp[2], label, sp[-2], content))
            except Exception as ex:
                traceback.print_exc(ex)

    aa = list(map(lambda x: (x.label_by_people, x.content), items))

    df = pd.DataFrame(data=aa)
    df.columns = ["label", "line"]

    bb = df.groupby(['label']).count()
    print(bb)

    save_file_path = "/Users/jiananliu/work/AISEALs/data/text_classification/tribe_labels_raw_data/new_finnal_data.txt"
    df.to_csv(save_file_path, index=False, sep="\t", header=False)


