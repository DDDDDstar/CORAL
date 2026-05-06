#!/bin/bash

# check if the dataset is provided
if [ -z "$1" ]; then
    echo "Please provide the dataset name with t2i-10M | laion-10M | webvid-2.5M | t2i-100M | t2i-1B"
    exit 1
fi

# check if the dataset is valid
# if [ "$1" != "t2i-1M" ] && [ "$1" != "t2i-10M" ] && [ "$1" != "t2i-30M" ] && [ "$1" != "t2i-40M" ] && [ "$1" != "t2i-50M" ] && [ "$1" != "laion-10M" ] && [ "$1" != "webvid-2.5M" ] && [ "$1" != "t2i-100M" ] && [ "$1" != "t2i-1B" ]; then
#     echo "Invalid dataset name in [t2i-10M, laion-10M, clip-webvid-2.5M, t2i-100M, t2i-1B]"
#     exit 1
# fi

prefix="./data"
dataset=${prefix}/$1

mkdir -p $prefix
mkdir -p $dataset

# download_partial_with_resume() {
#     local URL="$1"
#     local OUT="$2"
#     local MAX_END="$3"

#     if [ -z "$URL" ] || [ -z "$OUT" ] || [ -z "$MAX_END" ]; then
#         echo "Usage: download_partial_with_resume <url> <output_file> <max_end_byte>"
#         return 1
#     fi

#     local CUR_SIZE=0
#     if [ -f "$OUT" ]; then
#         CUR_SIZE=$(stat -c%s "$OUT")
#     fi

#     # 已完成
#     if [ "$CUR_SIZE" -gt "$MAX_END" ]; then
#         echo "[OK] File already complete: $OUT (size=$CUR_SIZE)"
#         return 0
#     fi

#     echo "[INFO] Downloading:"
#     echo "       URL      = $URL"
#     echo "       OUT      = $OUT"
#     echo "       RANGE    = ${CUR_SIZE}-${MAX_END}"

#     # curl -L \
#     #      -C "$CUR_SIZE" \
#     #      -r "${CUR_SIZE}-${MAX_END}" \
#     #      -o "$OUT" \
#     #      "$URL"
#     aria2c -c -x 8 -s 8 --min-split-size=50M --file-allocation=trunc \
#     --retry-wait=5 \
#     --max-tries=0 \
#     -o "$OUT" \
#     "$URL"

#     local ret=$?
#     if [ $ret -ne 0 ]; then
#         echo "[ERROR] curl failed with code $ret"
#         return $ret
#     fi

#     echo "[DONE] Download finished: $(stat -c%s "$OUT") bytes"
#     return 0
# }
download_partial_with_resume() {
    local URL="$1"
    local OUT="$2"
    local MAX_END="$3"

    if [ -z "$URL" ] || [ -z "$OUT" ] || [ -z "$MAX_END" ]; then
        echo "Usage: download_partial_with_resume <url> <output_file> <max_end_byte>"
        return 1
    fi

    if ! [[ "$MAX_END" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] MAX_END must be a non-negative integer"
        return 1
    fi

    local CUR_SIZE=0
    if [ -f "$OUT" ]; then
        CUR_SIZE=$(stat -c%s "$OUT")
    fi

    local TARGET_SIZE=$((MAX_END + 1))

    if [ "$CUR_SIZE" -ge "$TARGET_SIZE" ]; then
        echo "[OK] File already complete: $OUT (size=$CUR_SIZE, target=$TARGET_SIZE)"
        return 0
    fi

    echo "[INFO] Downloading:"
    echo "       URL      = $URL"
    echo "       OUT      = $OUT"
    echo "       RANGE    = ${CUR_SIZE}-${MAX_END}"

    curl -L \
         -r "${CUR_SIZE}-${MAX_END}" \
         -o "${OUT}.part" \
         "$URL"

    local ret=$?
    if [ $ret -ne 0 ]; then
        echo "[ERROR] curl failed with code $ret"
        rm -f "${OUT}.part"
        return $ret
    fi

    cat "${OUT}.part" >> "$OUT"
    rm -f "${OUT}.part"

    local NEW_SIZE
    NEW_SIZE=$(stat -c%s "$OUT")
    echo "[DONE] Download finished: $NEW_SIZE bytes"

    if [ "$NEW_SIZE" -gt "$TARGET_SIZE" ]; then
        echo "[WARN] File is larger than expected target size"
    fi

    return 0
}

if [ "$1" = "t2i-1M" ]; then
    echo "dataset t2i-1M"
    need_size=$((200*4*1000000+8-1))
    query_10k_size=$((200*4*10000+8-1))
    # download the dataset
    if [ ! -e ${dataset}/base.fbin ]; then
        curl -r 0-${need_size} -o ${dataset}/base.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.10M.fbin
        python change_data_in_file.py ${dataset}/base.fbin ${dataset}/base.fbin 1000000 200
    fi
    if [ ! -e $dataset/query.train.fbin ]; then
        curl -r 0-${need_size} -o $dataset/query.train.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin
        python change_data_in_file.py ${dataset}/query.train.fbin ${dataset}/query.train.fbin 100000 200
    fi
    if [ ! -e ${dataset}/query.fbin ]; then
        curl -r 0-${query_10k_size} -o ${dataset}/query.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin
        python change_data_in_file.py ${dataset}/query.fbin ${dataset}/query.fbin 10000 200
    fi

    if [ ! -e ${dataset}/query.gt.bin ]; then
        ../build/tests/comp_gt \
        --base_data_path ${dataset}/base.fbin  \
        --query_data_path ${dataset}/query.fbin  \
        --gt_save_file ${dataset}/gt.bin --k 100
    fi
    # if [ ! -e ${dataset}/train.gt.bin ]; then
    #     ../build/tests/comp_gt \
    #     --base_data_path ${dataset}/base.fbin  \
    #     --query_data_path ${dataset}/query.train.fbin  \
    #     --gt_save_file ${dataset}/train.gt.bin --k 100
    # fi
elif [ "$1" = "t2i-10M" ]; then
    echo "dataset t2i-10M"
    need_size=$((200*4*10000000+8-1))
    query_10k_size=$((200*4*10000+8-1))
    # download the dataset
    if [ ! -e ${dataset}/base.fbin ]; then
        curl -r 0-${need_size} -o ${dataset}/base.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.10M.fbin
    fi
    if [ ! -e ${dataset}/query.fbin ]; then
        curl -r 0-${query_10k_size} -o ${dataset}/query.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin
        python change_meta_data_in_file.py ${dataset}/query.fbin 10000
    fi
    need_size=$((200*4*5000000+8-1))
    if [ ! -e ${dataset}/query.train.fbin ]; then
        curl -r 0-${need_size} -o ${dataset}/query.train.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin
        python change_data_in_file.py ${dataset}/query.train.fbin ${dataset}/query.train.fbin 1000000 200
    fi

    if [ ! -e ${dataset}/query.gt.bin ]; then
        ../build/tests/comp_gt \
        --base_data_path ${dataset}/base.fbin  \
        --query_data_path ${dataset}/query.gt.bin  \
        --gt_save_file ${dataset}/query.gt.bin --k 100
    fi
    # if [ ! -e ${dataset}/train.gt.bin ]; then
    #     ../build/tests/comp_gt \
    #     --base_data_path ${dataset}/base.fbin  \
    #     --query_data_path ${dataset}/query.train.fbin  \
    #     --gt_save_file ${dataset}/train.gt.bin --k 100
    # fi
elif [ "$1" = "t2i-1B" ]; then
    echo "dataset t2i-1B"
    need_size=$((200*4*1000000000+8-1))
    query_10k_size=$((200*4*10000+8-1))
    URL="https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin"
    OUT="$dataset/base.fbin"
    download_partial_with_resume "$URL" "$OUT" "$need_size"

    URL="https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin"
    OUT="$dataset/query.fbin"
    download_partial_with_resume "$URL" "$OUT" "$query_10k_size"
    python change_meta_data_in_file.py $OUT 10000

    need_size=$((200*4*30000000+8-1))
    URL="https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin"
    OUT="$dataset/query.train.fbin"
    download_partial_with_resume "$URL" "$OUT" "$need_size"
    python3 change_meta_data_in_file.py $OUT 30000000

    if [ ! -e ${dataset}/query.gt.bin ]; then
        ../build/tests/comp_gt \
        --base_data_path ${dataset}/base.fbin  \
        --query_data_path ${dataset}/query.fbin  \
        --gt_save_file ${dataset}/query.gt.bin --k 100
    fi

    # if [ ! -e ${dataset}/train.gt.bin ]; then
    #     ../build/tests/comp_gt \
    #     --base_data_path ${dataset}/base.fbin  \
    #     --query_data_path ${dataset}/query.train.fbin  \
    #     --gt_save_file ${dataset}/train.gt.bin --k 100
    # fi
elif [ "$1" = "t2i-100M" ]; then
    echo "dataset t2i-100M"
    need_size=$((200*4*100000000+8-1))
    query_10k_size=$((200*4*10000+8-1))
    URL="https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin"
    OUT="$dataset/base.fbin"
    download_partial_with_resume "$URL" "$OUT" "$need_size"
    python change_meta_data_in_file.py $OUT 100000000
    
    if [ ! -e $dataset/query.fbin ]; then
        curl -r 0-${query_10k_size} -o $dataset/query.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin
        python change_meta_data_in_file.py $dataset/t2i-10M/query.fbin 10000
    fi
    
    need_size=$((200*4*10000000+8-1))
    URL="https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin"
    OUT="${dataset}/query.train.fbin"
    download_partial_with_resume "$URL" "$OUT" "$need_size"
    python change_meta_data_in_file.py $OUT 10000000

    if [ ! -e ${dataset}/query.gt.bin ]; then
        ../build/tests/comp_gt \
        --base_data_path ${dataset}/base.fbin  \
        --query_data_path ${dataset}/query.fbin  \
        --gt_save_file ${dataset}/query.gt.bin --k 100
    fi

    # if [ ! -e ${dataset}/train.gt.bin ]; then
    #     ../build/tests/comp_gt \
    #     --base_data_path ${dataset}/base.fbin  \
    #     --query_data_path ${dataset}/query.train.fbin  \
    #     --gt_save_file ${dataset}/train.gt.bin --k 100
    # fi
elif [ "$1" = "laion-10M" ]; then
    echo "dataset laion"
    # download the dataset
    for i in 0 1 2 3 4 5 6 7 9 10
    do
        if [ ! -e $dataset/img_emb_${i}.npy ]; then
            wget -c https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_${i}.npy -P $dataset
        fi
    done
    for i in 0 1 2 3 4 5 6 7 9 10
    do
        if [ ! -e $dataset/text_emb_${i}.npy ]; then
            wget -c https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/text_emb/text_emb_${i}.npy -P $dataset
        fi
    done

    # # export text and img simultaneously, watch out the DRAM.
    python3 export_fbin_from_npy.py
    python change_data_in_file.py ${dataset}/query.train.fbin ${dataset}/query.train.fbin 1000000 512
    if [ ! -e $dataset/gt.bin ]; then
        curl -o $dataset/query.gt.bin https://zenodo.org/records/11090378/files/laion.gt.10k.ibin
    fi
    if [ ! -e $dataset/query.fbin ]; then
        curl -o $dataset/query.fbin https://zenodo.org/records/11090378/files/laion.query.10k.fbin
    fi
    # if [ ! -e ${dataset}/train.gt.bin ]; then
    #     ../build/tests/comp_gt \
    #     --base_data_path ${dataset}/base.fbin  \
    #     --query_data_path ${dataset}/query.train.fbin  \
    #     --gt_save_file ${dataset}/train.gt.bin --k 100 --dist ip
    # fi
elif [ "$1" = "webvid-2.5M" ]; then
    echo "dataset clip-webvid"
    if [ ! -e $dataset/base.fbin ]; then
        wget -O $dataset/base.fbin https://zenodo.org/records/11090378/files/clip.webvid.base.2.5M.fbin
        # you can run prepare_for_clip_webvid on your own to generate base.2.5M.fbin.
        # mkdir -p ./data/clip-webvid-2.5M/temp_tar_data/
        # python3 prepare_for_clip_webvid.py
    fi

    if [ ! -e $dataset/query.train.2.5M.fbin ]; then
        curl -o $dataset/query.train.fbin https://zenodo.org/records/11090378/files/webvid.query.train.2.5M.fbin
        python change_data_in_file.py ${dataset}/query.train.fbin ${dataset}/query.train.fbin 1000000 512
    fi

    if [ ! -e $dataset/gt.bin ]; then
        curl -o $dataset/query.fbin https://zenodo.org/records/11090378/files/webvid.query.10k.fbin
        curl -o $dataset/query.gt.bin https://zenodo.org/records/11090378/files/webvid.gt.10k.ibin
    fi

    # if [ ! -e ${dataset}/train.gt.bin ]; then
    #     ../build/tests/comp_gt \
    #     --base_data_path ${dataset}/base.fbin  \
    #     --query_data_path ${dataset}/query.train.fbin  \
    #     --gt_save_file ${dataset}/train.gt.bin --k 100 --dist ip
    # fi
elif [ "$1" = "wit-1M" ]; then
    wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2c7a4f7c9d8f9f0b6d0f3d4/ViT-B-32.pt
    python wit.py --output_dir ${dataset}/wit_base_1m --target_size 1010000 --language none --split train --clip_model ./ViT-B-32.pt --batch_size 64 --skip_bad_file
    cp ${dataset}/wit_base_1m/base.fbin ${dataset}/base.fbin
    cp ${dataset}/wit_base_1m/query.fbin ${dataset}/query.fbin
    cp ${dataset}/wit_base_1m/query.fbin ${dataset}/query.train.fbin
    python change_data_in_file.py ${dataset}/query.train.fbin ${dataset}/query.train.fbin 100000 512
    python change_data_in_file.py ${dataset}/query.fbin ${dataset}/query.fbin 10000 512 100000
    
    # ../build/tests/comp_gt \
    #     --base_data_path ${dataset}/base.fbin  \
    #     --query_data_path ${dataset}/query.train.fbin  \
    #     --gt_save_file ${dataset}/train.gt.bin --k 100 --dist ip

    ../build/tests/comp_gt \
        --base_data_path ${dataset}/base.fbin  \
        --query_data_path ${dataset}/query.fbin  \
        --gt_save_file ${dataset}/query.gt.bin --k 100 --dist ip
elif [ "$1" = "cc3m-1M" ]; then
    wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2c7a4f7c9d8f9f0b6d0f3d4/ViT-B-32.pt
    python cc.py \
        --output_dir ${dataset}/cc3m_out \
        --clip_model ./iT-B-32.pt \
        --num_shards 250 \
        --target_size 1000000 \
        --device cuda 

    python cc.py \
        --output_dir ${dataset}/cc3m_out \
        --num_shards 3 \
        --mode test --test_size 10000 \
    
    cp ${dataset}/cc3m_out/base.fbin ${dataset}/base.fbin
    cp ${dataset}/cc3m_out/query.fbin ${dataset}/query.fbin
    cp ${dataset}/cc3m_out/query.train.fbin ${dataset}/query.train.fbin
    python change_data_in_file.py ${dataset}/query.train.fbin ${dataset}/query.train.fbin 100000 512
    python change_data_in_file.py ${dataset}/query.fbin ${dataset}/query.fbin 10000 512
        
    # ../build/tests/comp_gt \
    #     --base_data_path ${dataset}/base.fbin  \
    #     --query_data_path ${dataset}/query.train.fbin  \
    #     --gt_save_file ${dataset}/train.gt.bin --k 100 --dist ip

    ../build/tests/comp_gt \
        --base_data_path ${dataset}/base.fbin  \
        --query_data_path ${dataset}/query.fbin  \
        --gt_save_file ${dataset}/query.gt.bin --k 100 --dist ip
fi
