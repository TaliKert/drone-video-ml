#!/bin/bash

for filename in ./coco/labels/train2017/*.txt; do
  if grep -q '^0.*' $filename; then
    grep '^0.*' $filename > $(echo $filename | sed 's/labels/labels-people/g')
    echo $filename | sed 's/labels/images/g' | sed 's/txt/jpg/g' >> model/data/person-coco-train.txt
  fi
done
for filename in ./coco/labels/val2017/*.txt; do
  if grep -q '^0.*' $filename; then
    grep '^0.*' $filename > $(echo $filename | sed 's/labels/labels-people/g')
    echo $filename | sed 's/labels/images/g' | sed 's/txt/jpg/g' >> model/data/person-coco-valid.txt
  fi
done

for filename in ./coco/labels/train2017/*.txt; do
  cp $(echo $filename | sed 's/txt/jpg/g' | sed 's/labels/images/g') $(echo $filename | sed 's/txt/jpg/g' | sed 's/labels/images-people/g')
done
for filename in ./coco/labels/val2017/*.txt; do
  cp $(echo $filename | sed 's/txt/jpg/g' | sed 's/labels/images/g') $(echo $filename | sed 's/txt/jpg/g' | sed 's/labels/images-people/g')
done

mv coco/labels coco/labels-original
mv coco/images coco/images-original

mv coco/labels-people coco/labels
mv coco/images-people coco/images