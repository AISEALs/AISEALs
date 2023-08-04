#!/bin/bash
find /opt/tribe_labels/online/img_data/ -mtime +3 -exec rm {} \;
