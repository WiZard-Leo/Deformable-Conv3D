# 不含有center loss (baseline) 需要加入center loss dropout=0.3可能太大了
# python train.py --device cuda:0 \
#                 --batchsize 16 \
#                 --learning_rate 0.01 \
#                 --epochs 30 \
#                 --model small_deconv3d \
#                 --conf_thres 0.4 \
#                 --label_smooth 0.1 \
#                 --num_class 5 \
#                 --p_dropout 0.3 \
#                 --p_conv_dropout 0.3

# 含有center loss
python train.py --device cuda:0 \
                --batchsize 16 \
                --learning_rate 0.01 \
                --epochs 30 \
                --model small_deconv3d_centerloss \
                --center_loss \
                --conf_thres 0.4 \
                --label_smooth 0.1 \
                --num_class 5 \
                --p_dropout 0.1 \
                --p_conv_dropout 0.1

# # dropout 0.5
# python train.py --device cuda:0 \
#                 --batchsize 16 \
#                 --learning_rate 0.01 \
#                 --epochs 30 \
#                 --model small_deconv3d_dropout0.5 \
#                 --conf_thres 0.4 \
#                 --label_smooth 0.1 \
#                 --num_class 5 \
#                 --p_dropout 0.5 \
#                 --p_conv_dropout 0.5

# # learning rate 1e-3
# python train.py --device cuda:0 \
#                 --batchsize 16 \
#                 --learning_rate 0.001 \
#                 --epochs 30 \
#                 --model small_deconv3d_lr1e-3 \
#                 --conf_thres 0.4 \
#                 --label_smooth 0.1 \
#                 --num_class 5 \
#                 --p_dropout 0.3 \
#                 --p_conv_dropout 0.3
