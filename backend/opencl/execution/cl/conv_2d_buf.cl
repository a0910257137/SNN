#ifdef SNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

#define MOD_NUM 15


__kernel
void conv_2d_c4h1w4(GLOBAL_SIZE_2_DIMS
                    __global const FLOAT *input, 
#if(defined USE_LOW_BIT_WEIGHT_INT8)
                    __global const char *weight,
                    __global const FLOAT *dequantScale,
                    __global const FLOAT *dequantOffset,
#elif(defined USE_LOW_BIT_WEIGHT_INT4)
                    __global const uchar *weight,
                    __global const FLOAT *dequantScale,
                    __global const FLOAT *dequantOffset,
#else
                    __global const FLOAT *weight, 
#endif
                    __global const FLOAT *bias,
                    __global FLOAT *output,
                    __private const int2 in_hw,
                    __private const int inChannel, 
                    __private const int in_c_blocks,
                    __private const int2 out_hw,
                    __private const int2 filter_hw,
                    __private const int2 stride_hw,
                    __private const int2 pad_hw,
                    __private const int2 dilate_hw,
                    __private const int out_h_blocks,
                    __private const int out_w_blocks,
                    __private const int out_c_blocks
                    ) 
{
    const int out_c_w_idx = get_global_id(0); // c/4 w
    const int out_b_h_idx = get_global_id(1); // b h
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);
    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = (out_c_w_idx % out_w_blocks) << 2;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;

#if (defined USE_LOW_BIT_WEIGHT_INT4) || (defined USE_LOW_BIT_WEIGHT_INT4)
    const int FLOAT4 dequantScaleC4 = vload4(out_c_idx, dequantScale);
    const int FLOAT4 dequantOffsetC4 = vload4(out_c_idx, dequantOffset);
#endif
    FLOAT4 out0 = vload4(out_c_idx, bias);
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int in_w0_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_w1_idx_base = in_w0_idx_base + stride_hw.y;
    const int in_w2_idx_base = in_w1_idx_base + stride_hw.y;
    const int in_w3_idx_base = in_w2_idx_base + stride_hw.y;

    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + kh_start)*filter_hw.y + 0) * 4;

        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += dilate_hw.x) {
            const int inp_offset_base = (((out_b_idx * in_c_blocks + in_c_idx) * in_hw.x + iy) * in_hw.y + 0) * 4;

            for(int fw = 0; fw < filter_hw.y; fw++) {
                const int in_w0_idx = fw * dilate_hw.y + in_w0_idx_base;
                const int in_w1_idx = fw * dilate_hw.y + in_w1_idx_base;
                const int in_w2_idx = fw * dilate_hw.y + in_w2_idx_base;
                const int in_w3_idx = fw * dilate_hw.y + in_w3_idx_base;

                FLOAT4 in0 = (in_w0_idx < 0 || in_w0_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w0_idx, input+inp_offset_base);
                FLOAT4 in1 = (in_w1_idx < 0 || in_w1_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w1_idx, input+inp_offset_base);
                FLOAT4 in2 = (in_w2_idx < 0 || in_w2_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w2_idx, input+inp_offset_base);
                FLOAT4 in3 = (in_w3_idx < 0 || in_w3_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w3_idx, input+inp_offset_base);

#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(0, weight+weight_offset);
                char4 charWeight1 = vload4(0, weight+weight_offset+weight_oc_offset);
                char4 charWeight2 = vload4(0, weight+weight_offset+weight_oc_offset*2);
                char4 charWeight3 = vload4(0, weight+weight_offset+weight_oc_offset*3);
                FLOAT4 weight0 = mad(CONVERT_FLOAT4(charWeight0), dequantScaleC4, dequantOffsetC4);
                FLOAT4 weight1 = mad(CONVERT_FLOAT4(charWeight1), dequantScaleC4, dequantOffsetC4);
                FLOAT4 weight2 = mad(CONVERT_FLOAT4(charWeight2), dequantScaleC4, dequantOffsetC4);
                FLOAT4 weight3 = mad(CONVERT_FLOAT4(charWeight3), dequantScaleC4, dequantOffsetC4);
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                uchar2 charWeightInt40 = vload2(0, weight+weight_offset/2);
                uchar2 charWeightInt41 = vload2(0, weight+weight_offset/2+weight_oc_offset/2);
                uchar2 charWeightInt42 = vload2(0, weight+weight_offset/2+weight_oc_offset*2/2);
                uchar2 charWeightInt43 = vload2(0, weight+weight_offset/2+weight_oc_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                FLOAT4 weight0 = mad(CONVERT_FLOAT4(charWeight0), dequantScaleC4, dequantOffsetC4);
                FLOAT4 weight1 = mad(CONVERT_FLOAT4(charWeight1), dequantScaleC4, dequantOffsetC4);
                FLOAT4 weight2 = mad(CONVERT_FLOAT4(charWeight2), dequantScaleC4, dequantOffsetC4);
                FLOAT4 weight3 = mad(CONVERT_FLOAT4(charWeight3), dequantScaleC4, dequantOffsetC4);
#else
                FLOAT4 weight0 = vload4(0, weight+weight_offset);
                FLOAT4 weight1 = vload4(0, weight+weight_offset+weight_oc_offset);
                FLOAT4 weight2 = vload4(0, weight+weight_offset+weight_oc_offset*2);
                FLOAT4 weight3 = vload4(0, weight+weight_offset+weight_oc_offset*3);
#endif

                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
                out2 = mad(in2.x, weight0, out2);
                out2 = mad(in2.y, weight1, out2);
                out2 = mad(in2.z, weight2, out2);
                out2 = mad(in2.w, weight3, out2);
                
                out3 = mad(in3.x, weight0, out3);
                out3 = mad(in3.y, weight1, out3);
                out3 = mad(in3.z, weight2, out3);
                out3 = mad(in3.w, weight3, out3);
                
                weight_offset += 4;
            }
        }
    }
#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif
    const int out_offset = (((out_b_idx*out_c_blocks + out_c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;

#ifdef BLOCK_LEAVE
    const int remain = out_hw.y - out_w_idx;

    if (remain >= 4) {
        vstore16((FLOAT16)(out0, out1, out2, out3), 0, output+out_offset);
    }else if(remain == 3){
        vstore8((FLOAT8)(out0, out1), 0, output+out_offset);
        vstore4(out2, 2, output+out_offset);
    }else if(remain == 2){
        vstore8((FLOAT8)(out0, out1), 0, output+out_offset);
    }else if(remain == 1){
        vstore4(out0, 0, output+out_offset);
    }
#else
    vstore16((FLOAT16)(out0, out1, out2, out3), 0, output+out_offset);
#endif
}