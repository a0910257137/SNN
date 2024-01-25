#ifdef SNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define READ_INPUT_IMAGE(i, base)                                              \
  int inOffset##i = inWidthOffset##i + base;                                   \
  inOffset##i = select(inCurIdx + inOffset##i, -1,                             \
                       (inOffset##i < 0 || inOffset##i >= dw1_inputShape.y));  \
  inValue##i = RI_F(input, SAMPLER, (int2)(inOffset##i, inHeightIdx));

#define CALCULATE_OUTPUT(i)                                                    \
  outValue##i = mad(inValue##i.x, weights0, outValue##i);                      \
  outValue##i = mad(inValue##i.y, weights1, outValue##i);                      \
  outValue##i = mad(inValue##i.z, weights2, outValue##i);                      \
  outValue##i = mad(inValue##i.w, weights3, outValue##i);

#define CALCULATE_OUTPUT_WEIGHTS4(i, j)                                        \
  outValue##i = mad(inValue##j.x, weights4, outValue##i);                      \
  outValue##i = mad(inValue##j.y, weights5, outValue##i);                      \
  outValue##i = mad(inValue##j.z, weights6, outValue##i);                      \
  outValue##i = mad(inValue##j.w, weights7, outValue##i);

#define CALCULATE_OUTPUT_OPT(i)                                                \
  outValue##i = mad(in_sm##i[local_idx].x, weights0, outValue##i);             \
  outValue##i = mad(in_sm##i[local_idx].y, weights1, outValue##i);             \
  outValue##i = mad(in_sm##i[local_idx].z, weights2, outValue##i);             \
  outValue##i = mad(in_sm##i[local_idx].w, weights3, outValue##i);

#define GLOBAL_SIZE_2_DIMS                                                     \
  __private const int global_size_dim0, __private const int global_size_dim1,

__constant sampler_t SAMPLER =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                  \
  if (input1 >= global_size_dim0 || input2 >= global_size_dim1) {              \
    return;                                                                    \
  }

#define GLOBAL_SIZE_3_DIMS                                                     \
  __private const int global_size_dim0, __private const int global_size_dim1,  \
      __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                          \
  if (input1 >= global_size_dim0 || input2 >= global_size_dim1 ||              \
      input3 >= global_size_dim2) {                                            \
    return;                                                                    \
  }

#define UNIT 4
// convert data from buffer(nhwc) to conv to  image(b h, ic/4 w)
__kernel
#if SET_ATTRIBUTE
    __attribute__((work_group_size_hint(16, 16, 1)))
#endif
    void
    stem_conv_2d_c8h4w1(GLOBAL_SIZE_2_DIMS 
#ifdef BUFFER_IMAGE_IO_TRANS
                      __global const float *input_ptr,
#else
                      __global const FLOAT *input_ptr,
#endif
                      __private const int inHeight,
                      __private const int inWidth,
                      __private const int inChannels,
                      __read_only image2d_t weights,
#ifdef BIAS
                      __read_only image2d_t bias,
#endif
                      __private const int inChannelBlockLength,
                      __private const int2 outShapes,
                      __private const int2 weightShape,
                      __private const int2 strideShape,
                      __private const int2 paddingShape,
                      __private const int2 dilationShape,
                      __private const int outWidthBlocks,
                      __private const int outChannelBlocks,
                      __private const int outHeightBlocks,
                      __write_only image2d_t output)
    {
      // input tensor is 320 * 320 * 3
      const int outChannelWidthIdx =  get_global_id(0);
      
      const int outBatchHeightIdx =  get_global_id(1);
      
      DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outBatchHeightIdx);
      const int outChannelBlockIdx = (outChannelWidthIdx / outWidthBlocks) << 1;
      const int outWidthBlockIdx = outChannelWidthIdx % outWidthBlocks;
      const int outHeightBlockIdx = outBatchHeightIdx % outHeightBlocks;
      const int outBatchBlockIdx = outBatchHeightIdx / outHeightBlocks;
      
      #ifdef BIAS
        FLOAT4 outValue0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
        FLOAT4 outValue4 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx + 1, 0));
      #else
        FLOAT4 outValue0 = (FLOAT4)0;
        FLOAT4 outValue4 = (FLOAT4)0;
      #endif
        FLOAT4 outValue1 = outValue0;
        FLOAT4 outValue2 = outValue0;
        FLOAT4 outValue3 = outValue0;
        FLOAT4 outValue5 = outValue4;
        FLOAT4 outValue6 = outValue4;
        FLOAT4 outValue7 = outValue4;
      int inWidth0 = mad24(outWidthBlockIdx, strideShape.y, -paddingShape.y);
      int inHeight0 =
          mad24(outHeightBlockIdx, strideShape.x << 2, -paddingShape.x);
      int inHeight1 = inHeight0 + strideShape.x;
      int inHeight2 = inHeight1 + strideShape.x;
      int inHeight3 = inHeight2 + strideShape.x;
      int weightSize = mul24(weightShape.y, weightShape.x);
      const int weights_h_idx = mul24(outChannelBlockIdx, weightSize);
      const int batch_idx = mul24(outBatchBlockIdx, inHeight);
      FLOAT4 inValue0, inValue1, inValue2, inValue3;
      FLOAT4 weights0, weights1, weights2, weights3, weights4, weights5, weights6, weights7;
      int buffer_offset;
      #ifdef BUFFER_IMAGE_IO_TRANS
        __global float *input_current_ptr;
      #else
        __global FLOAT *input_current_ptr;
      #endif
      for (int inChannelBlockIdx = 0; inChannelBlockIdx < inChannelBlockLength; ++inChannelBlockIdx) {
          const int inIdx = mul24(inChannelBlockIdx, inWidth);
          int weights_x_idx = inChannelBlockIdx << 2;
          int weights_y_idx = weights_h_idx;
          for (int iy = 0; iy < weightShape.x * dilationShape.x;iy += dilationShape.x) {
          int h0 =
              select(inHeight0 + iy + batch_idx, -1,
                 (inHeight0 + iy < 0 || inHeight0 + iy >= inHeight));
          int h1 =
              select(inHeight1 + iy + batch_idx, -1,
                    (inHeight1 + iy < 0 || inHeight1 + iy >= inHeight));
          int h2 =
              select(inHeight2 + iy + batch_idx, -1,
                    (inHeight2 + iy < 0 || inHeight2 + iy >= inHeight));
          int h3 =
              select(inHeight3 + iy + batch_idx, -1,
                    (inHeight3 + iy < 0 || inHeight3 + iy >= inHeight));
          for (int ix = 0; ix < weightShape.y * dilationShape.y; ix += dilationShape.y) {
            int w0 = select(inWidth0 + ix + inIdx, -1,
                            (inWidth0 + ix < 0 || inWidth0 + ix >= inWidth));
            inValue0 = 0, inValue1 = 0,inValue2 = 0, inValue3 = 0; 
            #ifdef BUFFER_IMAGE_IO_TRANS
              // locate buffer positional index
              // value0
              // deal with boundary
              if( (h0 != -1) && (w0 != -1 ))
              {
                buffer_offset = ((batch_idx * inHeight + h0) * inWidth + w0) * inChannels;
                input_current_ptr = (input_ptr + buffer_offset);
                inValue0 = CONVERT_FLOAT4(vload4(0, input_current_ptr));
              }
              if( (h1 != -1) && (w0 != -1) )
              {
                // locate buffer positional index
                // value1
                buffer_offset = ((batch_idx * inHeight + h1) * inWidth + w0) * inChannels;
                input_current_ptr = (input_ptr + buffer_offset);
                inValue1 = CONVERT_FLOAT4(vload4(0, input_current_ptr));
              }
              if( (h2 != -1) && (w0 != -1) )
              {
                // locate buffer positional index
                // value2
                buffer_offset = ((batch_idx * inHeight + h2) * inWidth + w0) * inChannels ;
                input_current_ptr = (input_ptr + buffer_offset);
                inValue2 = CONVERT_FLOAT4(vload4(0, input_current_ptr));
              }
              if( (h3 != -1) && (w0 != -1) )
              {
                // locate buffer positional index
                // value3
                buffer_offset = ((batch_idx * inHeight + h3) * inWidth + w0) * inChannels;
                input_current_ptr = (input_ptr + buffer_offset);
                inValue3 = CONVERT_FLOAT4(vload4(0, input_current_ptr));
              }
            #else
              // locate buffer positional index
              // value0
              // deal with boundary
              if( (h0 != -1) && (w0 != -1) )
              {
                buffer_offset = ((batch_idx * inHeight + h0) * inWidth + w0) * inChannels;
                input_current_ptr = (input_ptr + buffer_offset);
                inValue0 = vload4(0, input_current_ptr);
              }
              if( (h1 != -1) && (w0 != -1) )
              {
                // locate buffer positional index
                // value1
                buffer_offset = ((batch_idx * inHeight + h1) * inWidth + w0) * inChannels;
                input_current_ptr = (input_ptr + buffer_offset);
                inValue1 = vload4(0, input_current_ptr);
              }
              if( (h2 != -1) && (w0 != -1) )
              {
                // locate buffer positional index
                // value2
                buffer_offset = ((batch_idx * inHeight + h2) * inWidth + w0) * inChannels;
                input_current_ptr = (input_ptr + buffer_offset);
                inValue2 = vload4(0, input_current_ptr);
              }
              if( (h3 != -1) && (w0 != -1) )
              {
                // locate buffer positional index
                // value3
                buffer_offset = ((batch_idx * inHeight + h3) * inWidth + w0) * inChannels;
                input_current_ptr = (input_ptr + buffer_offset);
                inValue3 = vload4(0, input_current_ptr);
              }
            #endif
            weights0 =
              RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weights_y_idx));
            weights1 =
                RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
            weights2 =
                RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
            weights3 =
                RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx));
            weights4 = RI_F(weights, SAMPLER,
                            (int2)(weights_x_idx + 0, weightSize + weights_y_idx));
            weights5 = RI_F(weights, SAMPLER,
                            (int2)(weights_x_idx + 1, weightSize + weights_y_idx));
            weights6 = RI_F(weights, SAMPLER,
                            (int2)(weights_x_idx + 2, weightSize + weights_y_idx));
            weights7 = RI_F(weights, SAMPLER,
                            (int2)(weights_x_idx + 3, weightSize + weights_y_idx++));
            CALCULATE_OUTPUT(0);
            CALCULATE_OUTPUT(1);
            CALCULATE_OUTPUT(2);
            CALCULATE_OUTPUT(3);
            CALCULATE_OUTPUT_WEIGHTS4(4, 0);
            CALCULATE_OUTPUT_WEIGHTS4(5, 1);
            CALCULATE_OUTPUT_WEIGHTS4(6, 2);
            CALCULATE_OUTPUT_WEIGHTS4(7, 3);
           }
         }
       }
      #ifdef RELU
        outValue0 = fmax(outValue0, (FLOAT4)0);
        outValue1 = fmax(outValue1, (FLOAT4)0);
        outValue2 = fmax(outValue2, (FLOAT4)0);
        outValue3 = fmax(outValue3, (FLOAT4)0);
        outValue4 = fmax(outValue4, (FLOAT4)0);
        outValue5 = fmax(outValue5, (FLOAT4)0);
        outValue6 = fmax(outValue6, (FLOAT4)0);
        outValue7 = fmax(outValue7, (FLOAT4)0);
      #endif

      #ifdef RELU6
        outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
        outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
        outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
        outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
        outValue4 = clamp(outValue4, (FLOAT4)0, (FLOAT4)6);
        outValue5 = clamp(outValue5, (FLOAT4)0, (FLOAT4)6);
        outValue6 = clamp(outValue6, (FLOAT4)0, (FLOAT4)6);
        outValue7 = clamp(outValue7, (FLOAT4)0, (FLOAT4)6);
      #endif
      const int out_x_base = mul24(outChannelBlockIdx, outShapes.y);
      const int out_y_base = mul24(outBatchBlockIdx, outShapes.x);
      int out_x_idx = outWidthBlockIdx;
      int out_y_idx = outHeightBlockIdx << 2;
      const int remain_y = outShapes.x - out_y_idx;
      int output_idx = out_x_base + out_x_idx;
      int output_idy = out_y_base + out_y_idx;
      if (remain_y >= 4) {
        WI_F(output, (int2)(output_idx, output_idy), outValue0);
        WI_F(output, (int2)(output_idx, output_idy + 1), outValue1);
        WI_F(output, (int2)(output_idx, output_idy + 2), outValue2);
        WI_F(output, (int2)(output_idx, output_idy + 3), outValue3);
      } else if (remain_y == 3) {
        WI_F(output, (int2)(output_idx, output_idy), outValue0);
        WI_F(output, (int2)(output_idx, output_idy + 1), outValue1);
        WI_F(output, (int2)(output_idx, output_idy + 2), outValue2);
      } else if (remain_y == 2) {
        WI_F(output, (int2)(output_idx, output_idy), outValue0);
        WI_F(output, (int2)(output_idx, output_idy + 1), outValue1);
      } else if (remain_y == 1) {
        WI_F(output, (int2)(output_idx, output_idy), outValue0);
      }
      if (outChannelBlockIdx + 1 >= outChannelBlocks) {
        return;
      }
      output_idx += outShapes.y;
      if (remain_y >= 4) {
        WI_F(output, (int2)(output_idx, output_idy), outValue4);
        WI_F(output, (int2)(output_idx, output_idy + 1), outValue5);
        WI_F(output, (int2)(output_idx, output_idy + 2), outValue6);
        WI_F(output, (int2)(output_idx, output_idy + 3), outValue7);
      } else if (remain_y == 3) {
        WI_F(output, (int2)(output_idx, output_idy), outValue4);
        WI_F(output, (int2)(output_idx, output_idy + 1), outValue5);
        WI_F(output, (int2)(output_idx, output_idy + 2), outValue6);
      } else if (remain_y == 2) {
        WI_F(output, (int2)(output_idx, output_idy), outValue4);
        WI_F(output, (int2)(output_idx, output_idy + 1), outValue5);
      } else if (remain_y == 1) {
        WI_F(output, (int2)(output_idx, output_idy), outValue4);
      }
    }

__kernel
#if SET_ATTRIBUTE
    __attribute__((work_group_size_hint(16, 16, 1)))
#endif
    void
    dw1_s11k33_conv2_s11k11_conv3_s11k11(
        GLOBAL_SIZE_2_DIMS 
        __private const int4 transChannelShape,
        __read_only image2d_t input,
        __write_only image2d_t output,
        __read_only image2d_t dw1_filter,
#ifndef NO_BIAS
        __read_only image2d_t dw1_bias,
#endif
        __private const int2 dw1_inputShape,
        __private const int dw1_inChannelBlocks,
        __private const int2 dw1_outputShape,
        __private const int2 dw1_filterShape,
        __private const int2 dw1_paddingShape,
        __read_only image2d_t conv2_filter,
#ifndef NO_BIAS
        __read_only image2d_t conv2_bias,
#endif  
        __private const int2 conv2_inputShape,
        __private const int2 conv2_outputShape,
        __private const int2 conv2_strideShape,
        __private const int conv2_outWidth4,
        __read_only image2d_t conv3_filter,
#ifndef NO_BIAS
        __read_only image2d_t conv3_bias,
#endif  
        __private const int2 conv3_inputShape,
        __private const int2 conv3_outputShape,
        __private const int2 conv3_strideShape)
        {
          const int outWidthIdx = get_global_id(0);
          const int outBatchHeightIdx = get_global_id(1);
          DEAL_NON_UNIFORM_DIM2(outWidthIdx, outBatchHeightIdx);
          int ow4 = (dw1_outputShape.y + 3) / 4;
          const int outWidthBlockidx = outWidthIdx % ow4;
          int inCurIdx;
          int dw1_numOutChannels = transChannelShape.x/4, conv2_numOutChannels = transChannelShape.y/4, conv3_numOutChannels = transChannelShape.z/4; 
          #ifdef DW1_TEMP8
              FLOAT4 dw1_tempWidthOffset0[8], dw1_tempWidthOffset1[8], dw1_tempWidthOffset2[8], dw1_tempWidthOffset3[8]; 
          #endif
          #ifdef DW1_TEMP12
              FLOAT4 dw1_tempWidthOffset0[12], dw1_tempWidthOffset1[12], dw1_tempWidthOffset2[12], dw1_tempWidthOffset3[12]; 
          #endif
          #ifdef DW1_TEMP16
              FLOAT4 dw1_tempWidthOffset0[16], dw1_tempWidthOffset1[16], dw1_tempWidthOffset2[16], dw1_tempWidthOffset3[16]; 
          #endif
          #ifdef DW1_TEMP20
              FLOAT4 dw1_tempWidthOffset0[20], dw1_tempWidthOffset1[20], dw1_tempWidthOffset2[20], dw1_tempWidthOffset3[20]; 
          #endif
          #ifdef DW1_TEMP24
              FLOAT4 dw1_tempWidthOffset0[24], dw1_tempWidthOffset1[24], dw1_tempWidthOffset2[24], dw1_tempWidthOffset3[24]; 
          #endif
          #ifdef DW1_TEMP28
              FLOAT4 dw1_tempWidthOffset0[28], dw1_tempWidthOffset1[28], dw1_tempWidthOffset2[28], dw1_tempWidthOffset3[28]; 
          #endif
          #ifdef DW1_TEMP32
              FLOAT4 dw1_tempWidthOffset0[32], dw1_tempWidthOffset1[32], dw1_tempWidthOffset2[32], dw1_tempWidthOffset3[32]; 
          #endif
          #ifdef DW1_TEMP48
              FLOAT4 dw1_tempWidthOffset0[48], dw1_tempWidthOffset1[48], dw1_tempWidthOffset2[48], dw1_tempWidthOffset3[48]; 
          #endif
          #ifdef DW1_TEMP64
              FLOAT4 dw1_tempWidthOffset0[64], dw1_tempWidthOffset1[64], dw1_tempWidthOffset2[64], dw1_tempWidthOffset3[64]; 
          #endif
          #ifdef DW1_TEMP96
              FLOAT4 dw1_tempWidthOffset0[96], dw1_tempWidthOffset1[96], dw1_tempWidthOffset2[96], dw1_tempWidthOffset3[96]; 
          #endif
          #ifdef DW1_TEMP240
              FLOAT4 dw1_tempWidthOffset0[240], dw1_tempWidthOffset1[240], dw1_tempWidthOffset2[240], dw1_tempWidthOffset3[240]; 
          #endif
          
          FLOAT4 inValue0, inValue1, inValue2, inValue3;
          FLOAT4 outValue0, outValue1, outValue2, outValue3; 
          FLOAT4 weights, weights0, weights1, weights2, weights3;
          int outChannelBlockIdx, inChannelBlockIdx;
          int outWidthBlockidx4 = outWidthBlockidx << 2;
          int inWidthOffset0 = outWidthBlockidx4 - dw1_paddingShape.y;
          int inWidthOffset1 = inWidthOffset0 + 1;
          int inWidthOffset2 = inWidthOffset0 + 2;
          int inWidthOffset3 = inWidthOffset0 + 3;
          const int outBatchIdx = mul24((outBatchHeightIdx / dw1_outputShape.x), dw1_inputShape.x);
          // Start depthwise 1x1
          for(int outChannelBlockIdx = 0;outChannelBlockIdx < dw1_numOutChannels; outChannelBlockIdx++)
          {
            int heightIdx = outBatchHeightIdx % dw1_outputShape.x - dw1_paddingShape.x;
            inChannelBlockIdx = outChannelBlockIdx;
            outValue0 = RI_F(dw1_bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
            outValue1 = outValue0;
            outValue2 = outValue0;
            outValue3 = outValue0;
            inCurIdx = mul24(inChannelBlockIdx, dw1_inputShape.y);
            const int inWidthIdx0 =
              select(inCurIdx + inWidthOffset0, -1,
                    (inWidthOffset0 < 0 || inWidthOffset0 >= dw1_inputShape.y));
            const int inWidthIdx1 =
              select(inCurIdx + inWidthOffset1, -1,
                      (inWidthOffset1 < 0 || inWidthOffset1 >= dw1_inputShape.y));
            const int inWidthIdx2 =
              select(inCurIdx + inWidthOffset2, -1,
                      (inWidthOffset2 < 0 || inWidthOffset2 >= dw1_inputShape.y));
            for (int kh = 0; kh < dw1_filterShape.x; kh++) {
              int inHeightIdx = select(heightIdx + outBatchIdx, -1,
                                      (heightIdx < 0 || heightIdx >= dw1_inputShape.x));
              heightIdx++;
              inValue1 = RI_F(input, SAMPLER, (int2)(inWidthIdx0, inHeightIdx));
              inValue2 = RI_F(input, SAMPLER, (int2)(inWidthIdx1, inHeightIdx));
              inValue3 = RI_F(input, SAMPLER, (int2)(inWidthIdx2, inHeightIdx)); 
              for (int kw = 0; kw < dw1_filterShape.y; kw++) {
                int filterIdx = mad24(kh, dw1_filterShape.y, kw);
                inValue0 = inValue1;
                inValue1 = inValue2;
                inValue2 = inValue3;
                int inWidthIdx = inWidthOffset3 + kw;
                inWidthIdx = select(inCurIdx + inWidthIdx, -1,
                                    (inWidthIdx < 0 || inWidthIdx >= dw1_inputShape.y));
                inValue3 = RI_F(input, SAMPLER, (int2)(inWidthIdx, inHeightIdx));
                weights =
                    RI_F(dw1_filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));
                outValue0 = mad(inValue0, weights, outValue0);
                outValue1 = mad(inValue1, weights, outValue1);
                outValue2 = mad(inValue2, weights, outValue2);
                outValue3 = mad(inValue3, weights, outValue3);
              }
            }
            #ifdef DW1_RELU
              outValue0 = fmax(outValue0, (FLOAT4)0);
              outValue1 = fmax(outValue1, (FLOAT4)0);
              outValue2 = fmax(outValue2, (FLOAT4)0);
              outValue3 = fmax(outValue3, (FLOAT4)0);
            #endif
            #ifdef DW1_RELU6
              outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
              outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
              outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
              outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
            #endif
            #ifdef DW1_SIGMOID
              outValue0 = native_recip(1.0f + (FLOAT4)native_exp(-outValue0));
              outValue1 = native_recip(1.0f + (FLOAT4)native_exp(-outValue1));
              outValue2 = native_recip(1.0f + (FLOAT4)native_exp(-outValue2));
              outValue3 = native_recip(1.0f + (FLOAT4)native_exp(-outValue3));
            #endif

            dw1_tempWidthOffset0[outChannelBlockIdx] = outValue0;
            dw1_tempWidthOffset1[outChannelBlockIdx] = outValue1;
            dw1_tempWidthOffset2[outChannelBlockIdx] = outValue2;
            dw1_tempWidthOffset3[outChannelBlockIdx] = outValue3;
            // const int remain = dw1_outputShape.y - outWidthBlockidx4;
            // int outWidthIdx =
            //     mul24(outChannelBlockIdx, dw1_outputShape.y) + outWidthBlockidx4;
            // if (remain >= 4) {
            //   WI_F(output, (int2)(outWidthIdx, outBatchHeightIdx), outValue0);
            //   WI_F(output, (int2)(outWidthIdx + 1, outBatchHeightIdx), outValue1);
            //   WI_F(output, (int2)(outWidthIdx + 2, outBatchHeightIdx), outValue2);
            //   WI_F(output, (int2)(outWidthIdx + 3, outBatchHeightIdx), outValue3);
            // } 
            // else if (remain == 3) {
            //   WI_F(output, (int2)(outWidthIdx, outBatchHeightIdx), outValue0);
            //   WI_F(output, (int2)(outWidthIdx + 1, outBatchHeightIdx), outValue1);
            //   WI_F(output, (int2)(outWidthIdx + 2, outBatchHeightIdx), outValue2);
            // } 
            // else if (remain == 2) {
            //   WI_F(output, (int2)(outWidthIdx, outBatchHeightIdx), outValue0);
            //   WI_F(output, (int2)(outWidthIdx + 1, outBatchHeightIdx), outValue1);
            // } 
            // else if (remain == 1) {
            //   WI_F(output, (int2)(outWidthIdx, outBatchHeightIdx), outValue0);
            // }
          }
          // Start convolution 1x1
          int conv2_batchIndex = outBatchHeightIdx / conv2_outputShape.x;
          // int input_height_block_idx =
          //     mul24((output_batch_height_idx % output_shape.x), stride_shape.x) +
          //     batch_index * input_shape.x;
          int conv2_numInChannels = transChannelShape.x / 4;
          // // conv output values
          #ifdef CONV2_TEMP4
              FLOAT4 conv2_tempWidthOffset0[4], conv2_tempWidthOffset1[4], conv2_tempWidthOffset2[4], conv2_tempWidthOffset3[4]; 
          #endif
          #ifdef CONV2_TEMP8
              FLOAT4 conv2_tempWidthOffset0[8], conv2_tempWidthOffset1[8], conv2_tempWidthOffset2[8], conv2_tempWidthOffset3[8]; 
          #endif
          #ifdef CONV2_TEMP12
              FLOAT4 conv2_tempWidthOffset0[12], conv2_tempWidthOffset1[12], conv2_tempWidthOffset2[12], conv2_tempWidthOffset3[12]; 
          #endif
          #ifdef CONV2_TEMP16
              FLOAT4 conv2_tempWidthOffset0[16], conv2_tempWidthOffset1[16], conv2_tempWidthOffset2[16], conv2_tempWidthOffset3[16];
          #endif
          #ifdef CONV2_TEMP20
              FLOAT4 conv2_tempWidthOffset0[20], conv2_tempWidthOffset1[20], conv2_tempWidthOffset2[20], conv2_tempWidthOffset3[20]; 
          #endif
          #ifdef CONV2_TEMP24
              FLOAT4 conv2_tempWidthOffset0[24], conv2_tempWidthOffset1[24], conv2_tempWidthOffset2[24], conv2_tempWidthOffset3[24]; 
          #endif
          #ifdef CONV2_TEMP28
              FLOAT4 conv2_tempWidthOffset0[28], conv2_tempWidthOffset1[28], conv2_tempWidthOffset2[28], conv2_tempWidthOffset3[28]; 
          #endif
          #ifdef CONV2_TEMP32
              FLOAT4 conv2_tempWidthOffset0[32], conv2_tempWidthOffset1[32], conv2_tempWidthOffset2[32], conv2_tempWidthOffset3[32]; 
          #endif
          #ifdef CONV2_TEMP48
              FLOAT4 conv2_tempWidthOffset0[48], conv2_tempWidthOffset1[48], conv2_tempWidthOffset2[48], conv2_tempWidthOffset3[48]; 
          #endif
          #ifdef CONV2_TEMP64
              FLOAT4 conv2_tempWidthOffset0[64], conv2_tempWidthOffset1[64], conv2_tempWidthOffset2[64], conv2_tempWidthOffset3[64];
          #endif
          #ifdef CONV2_TEMP96
              FLOAT4 conv2_tempWidthOffset0[96], conv2_tempWidthOffset1[96], conv2_tempWidthOffset2[96], conv2_tempWidthOffset3[96]; 
          #endif
          
          for(int conv2_outChannelBlockIdx = 0;conv2_outChannelBlockIdx < conv2_numOutChannels; conv2_outChannelBlockIdx++)
          {
            outValue0 = RI_F(conv2_bias, SAMPLER, (int2)(conv2_outChannelBlockIdx, 0));
            outValue1 = outValue0;
            outValue2 = outValue0;
            outValue3 = outValue0;
            // in##i
            for(int conv2_inChannelBlockidx = 0; conv2_inChannelBlockidx < conv2_numInChannels;
                ++conv2_inChannelBlockidx)
            {
              int conv2_inWidthBase = conv2_inChannelBlockidx * conv2_inputShape.y;
              int conv2_FilterWidthBase = conv2_inChannelBlockidx << 2;
              weights0 = RI_F(conv2_filter, SAMPLER, 
                              (int2)(conv2_FilterWidthBase + 0, conv2_outChannelBlockIdx));
              weights1 = RI_F(conv2_filter, SAMPLER, 
                              (int2)(conv2_FilterWidthBase + 1, conv2_outChannelBlockIdx));
              weights2 = RI_F(conv2_filter, SAMPLER, 
                              (int2)(conv2_FilterWidthBase + 2, conv2_outChannelBlockIdx));
              weights3 = RI_F(conv2_filter, SAMPLER, 
                              (int2)(conv2_FilterWidthBase + 3, conv2_outChannelBlockIdx));
              inValue0 = dw1_tempWidthOffset0[conv2_inChannelBlockidx]; 
              inValue1 = dw1_tempWidthOffset1[conv2_inChannelBlockidx];
              inValue2 = dw1_tempWidthOffset2[conv2_inChannelBlockidx];
              inValue3 = dw1_tempWidthOffset3[conv2_inChannelBlockidx];
              CALCULATE_OUTPUT(0);
              CALCULATE_OUTPUT(1);
              CALCULATE_OUTPUT(2);
              CALCULATE_OUTPUT(3);
            }
            #ifdef CONV2_RELU
              outValue0 = fmax(outValue0, (FLOAT4)0);
              outValue1 = fmax(outValue1, (FLOAT4)0);
              outValue2 = fmax(outValue2, (FLOAT4)0);
              outValue3 = fmax(outValue3, (FLOAT4)0);
            #endif
            #ifdef CONV2_RELU6
              outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
              outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
              outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
              outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
            #endif
            #ifdef CONV2_SIGMOID
              outValue0 = native_recip(1.0f + (FLOAT4)native_exp(-outValue0));
              outValue1 = native_recip(1.0f + (FLOAT4)native_exp(-outValue1));
              outValue2 = native_recip(1.0f + (FLOAT4)native_exp(-outValue2));
              outValue3 = native_recip(1.0f + (FLOAT4)native_exp(-outValue3));
            #endif
            conv2_tempWidthOffset0[conv2_outChannelBlockIdx] = outValue0;
            conv2_tempWidthOffset1[conv2_outChannelBlockIdx] = outValue1;
            conv2_tempWidthOffset2[conv2_outChannelBlockIdx] = outValue2;
            conv2_tempWidthOffset3[conv2_outChannelBlockIdx] = outValue3;
            // const int out_x_base = mul24(conv1_outChannelBlockIdx, conv1_outputShape.y);
            // int out_x_idx = outWidthBlockidx << 2;
            // const int remain = conv1_outputShape.y - out_x_idx;
            // int output_idx = out_x_base + out_x_idx;
            // if(remain >= 4)
            // { 
            //   WI_F(output, (int2)(output_idx, outBatchHeightIdx), outValue0);
            //   WI_F(output, (int2)(output_idx + 1, outBatchHeightIdx), outValue1);
            //   WI_F(output, (int2)(output_idx + 2, outBatchHeightIdx), outValue2);
            //   WI_F(output, (int2)(output_idx + 3, outBatchHeightIdx), outValue3);
            // }else if(remain == 3)
            // {
            //   WI_F(output, (int2)(output_idx, outBatchHeightIdx), outValue0);
            //   WI_F(output, (int2)(output_idx + 1, outBatchHeightIdx), outValue1);
            //   WI_F(output, (int2)(output_idx + 2, outBatchHeightIdx), outValue2);
            // }else if(remain == 2)
            // {
            //   WI_F(output, (int2)(output_idx, outBatchHeightIdx), outValue0);
            //   WI_F(output, (int2)(output_idx + 1, outBatchHeightIdx), outValue1);
            // }else if(remain == 1)
            // {
            //   WI_F(output, (int2)(output_idx, outBatchHeightIdx), outValue0);
            // }
          }
          // // Start convolution 1x1
          int conv3_numInChannels = transChannelShape.y / 4;
          for(int conv3_outChannelBlockIdx = 0;conv3_outChannelBlockIdx < conv3_numOutChannels; conv3_outChannelBlockIdx++)
          {
            outValue0 = RI_F(conv3_bias, SAMPLER, (int2)(conv3_outChannelBlockIdx, 0));
            outValue1 = outValue0;
            outValue2 = outValue0;
            outValue3 = outValue0;
            // in##i
            for(int conv3_inChannelBlockidx = 0; conv3_inChannelBlockidx < conv3_numInChannels;
                ++conv3_inChannelBlockidx)
            {
              int conv3_inWidthBase = conv3_inChannelBlockidx * conv3_inputShape.y;
              int conv3_FilterWidthBase = conv3_inChannelBlockidx << 2;
              weights0 = RI_F(conv3_filter, SAMPLER, 
                              (int2)(conv3_FilterWidthBase + 0, conv3_outChannelBlockIdx));
              weights1 = RI_F(conv3_filter, SAMPLER, 
                              (int2)(conv3_FilterWidthBase + 1, conv3_outChannelBlockIdx));
              weights2 = RI_F(conv3_filter, SAMPLER, 
                              (int2)(conv3_FilterWidthBase + 2, conv3_outChannelBlockIdx));
              weights3 = RI_F(conv3_filter, SAMPLER, 
                              (int2)(conv3_FilterWidthBase + 3, conv3_outChannelBlockIdx));
              
              inValue0 = conv2_tempWidthOffset0[conv3_inChannelBlockidx]; 
              inValue1 = conv2_tempWidthOffset1[conv3_inChannelBlockidx];
              inValue2 = conv2_tempWidthOffset2[conv3_inChannelBlockidx];
              inValue3 = conv2_tempWidthOffset3[conv3_inChannelBlockidx];
              
              CALCULATE_OUTPUT(0);
              CALCULATE_OUTPUT(1);
              CALCULATE_OUTPUT(2);
              CALCULATE_OUTPUT(3);
            }
            #ifdef CONV3_RELU
              outValue0 = fmax(outValue0, (FLOAT4)0);
              outValue1 = fmax(outValue1, (FLOAT4)0);
              outValue2 = fmax(outValue2, (FLOAT4)0);
              outValue3 = fmax(outValue3, (FLOAT4)0);
            #endif
            #ifdef CONV3_RELU6
              outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
              outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
              outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
              outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
            #endif
            #ifdef CONV3_SIGMOID
              outValue0 = native_recip(1.0f + (FLOAT4)native_exp(-outValue0));
              outValue1 = native_recip(1.0f + (FLOAT4)native_exp(-outValue1));
              outValue2 = native_recip(1.0f + (FLOAT4)native_exp(-outValue2));
              outValue3 = native_recip(1.0f + (FLOAT4)native_exp(-outValue3));
            #endif
            const int out_x_base = mul24(conv3_outChannelBlockIdx, conv3_outputShape.y);
            int out_x_idx = outWidthBlockidx << 2;
            const int remain = conv3_outputShape.y - out_x_idx;
            // printf("%d\n", remain);
            int output_idx = out_x_base + out_x_idx;
            if(remain >= 4){
              WI_F(output, (int2)(output_idx, outBatchHeightIdx), outValue0);
              WI_F(output, (int2)(output_idx + 1, outBatchHeightIdx), outValue1);
              WI_F(output, (int2)(output_idx + 2, outBatchHeightIdx), outValue2);
              WI_F(output, (int2)(output_idx + 3, outBatchHeightIdx), outValue3);
            }else if(remain == 3){
              WI_F(output, (int2)(output_idx, outBatchHeightIdx), outValue0);
              WI_F(output, (int2)(output_idx + 1, outBatchHeightIdx), outValue1);
              WI_F(output, (int2)(output_idx + 2, outBatchHeightIdx), outValue2);
            }else if(remain == 2){
              WI_F(output, (int2)(output_idx, outBatchHeightIdx), outValue0);
              WI_F(output, (int2)(output_idx + 1, outBatchHeightIdx), outValue1);
            }else if(remain == 1){
              WI_F(output, (int2)(output_idx, outBatchHeightIdx), outValue0);
            }
          }
}

__kernel
#if SET_ATTRIBUTE
    __attribute__((work_group_size_hint(16, 16, 1)))
#endif
    void
    dw_s11k33_dw_s11k11_dw_s11k11(
        GLOBAL_SIZE_2_DIMS 
        __private const int4 transChannelShape,
        __read_only image2d_t input,
        __write_only image2d_t output,
        __read_only image2d_t dw1_filter,
#ifndef NO_BIAS
        __read_only image2d_t dw1_bias,
#endif
        __private const int2 dw1_inputShape,
        __private const int dw1_inChannelBlocks,
        __private const int2 dw1_outputShape,
        __private const int2 dw1_filterShape,
        __private const int2 dw1_paddingShape,
        __read_only image2d_t dw2_filter,
#ifndef NO_BIAS
        __read_only image2d_t dw2_bias,
#endif
        __read_only image2d_t dw3_filter,
#ifndef NO_BIAS
        __read_only image2d_t dw3_bias
#endif
)
        {
          const int outChannelWidthIdx = get_global_id(0);
          const int outHeightBlockIdx = get_global_id(1);
          DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightBlockIdx);
          int ow4 = (dw1_outputShape.y + 3) / 4;
          const int outChannelBlockIdx = outChannelWidthIdx / ow4;
          const int outWidthBlockidx = outChannelWidthIdx % ow4;
          const int inChannelBlockIdx = outChannelBlockIdx;
          
        #ifndef NO_BIAS
          FLOAT4 outValue0 = RI_F(dw1_bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
        #else
          FLOAT4 outValue0 = (FLOAT4)(0.0f);
        #endif
          FLOAT4 outValue1 = outValue0;
          FLOAT4 outValue2 = outValue0;
          FLOAT4 outValue3 = outValue0;

          const int outWidthBlockidx4 = outWidthBlockidx << 2;
          const int inWidthOffset0 = outWidthBlockidx4 - dw1_paddingShape.y;
          const int inWidthOffset1 = inWidthOffset0 + 1;
          const int inWidthOffset2 = inWidthOffset0 + 2;
          const int inWidthOffset3 = inWidthOffset0 + 3;
          FLOAT4 weights;
          int heightIdx = outHeightBlockIdx % dw1_outputShape.x - dw1_paddingShape.x;
          const int outBatchIdx =
              mul24((outHeightBlockIdx / dw1_outputShape.x), dw1_inputShape.x);
          const int inCurIdx = mul24(inChannelBlockIdx, dw1_inputShape.y);

          const int inWidthIdx0 =
              select(inCurIdx + inWidthOffset0, -1,
                    (inWidthOffset0 < 0 || inWidthOffset0 >= dw1_inputShape.y));
          const int inWidthIdx1 =
              select(inCurIdx + inWidthOffset1, -1,
                    (inWidthOffset1 < 0 || inWidthOffset1 >= dw1_inputShape.y));
          const int inWidthIdx2 =
              select(inCurIdx + inWidthOffset2, -1,
                    (inWidthOffset2 < 0 || inWidthOffset2 >= dw1_inputShape.y));
          FLOAT4 inValue0, inValue1, inValue2, inValue3;
          for (int kh = 0; kh < dw1_filterShape.x; kh++) {
            int inHeightIdx = select(heightIdx + outBatchIdx, -1,
                                    (heightIdx < 0 || heightIdx >= dw1_inputShape.x));
            heightIdx++;
            inValue1 = RI_F(input, SAMPLER, (int2)(inWidthIdx0, inHeightIdx));
            inValue2 = RI_F(input, SAMPLER, (int2)(inWidthIdx1, inHeightIdx));
            inValue3 = RI_F(input, SAMPLER, (int2)(inWidthIdx2, inHeightIdx));
            for (int kw = 0; kw < dw1_filterShape.y; kw++) {
              int filterIdx = mad24(kh, dw1_filterShape.y, kw);
              inValue0 = inValue1;
              inValue1 = inValue2;
              inValue2 = inValue3;
              int inWidthIdx = inWidthOffset3 + kw;
              inWidthIdx = select(inCurIdx + inWidthIdx, -1,
                                  (inWidthIdx < 0 || inWidthIdx >= dw1_inputShape.y));
              inValue3 = RI_F(input, SAMPLER, (int2)(inWidthIdx, inHeightIdx));
              weights = RI_F(dw1_filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));
              outValue0 = mad(inValue0, weights, outValue0);
              outValue1 = mad(inValue1, weights, outValue1);
              outValue2 = mad(inValue2, weights, outValue2);
              outValue3 = mad(inValue3, weights, outValue3);
            }
          }

        #ifdef DW1_RELU
          outValue0 = fmax(outValue0, (FLOAT4)0);
          outValue1 = fmax(outValue1, (FLOAT4)0);
          outValue2 = fmax(outValue2, (FLOAT4)0);
          outValue3 = fmax(outValue3, (FLOAT4)0);
        #endif

        #ifdef DW1_RELU6
          outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
          outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
          outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
          outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
        #endif

          // const int remain = dw1_outputShape.y - outWidthBlockidx4;
          // int outWidthIdx =
          //     mul24(outChannelBlockIdx, dw1_outputShape.y) + outWidthBlockidx4;
          // if (remain >= 4) {
          //   WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
          //   WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
          //   WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
          //   WI_F(output, (int2)(outWidthIdx + 3, outHeightBlockIdx), outValue3);
          // } else if (remain == 3) {
          //   WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
          //   WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
          //   WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
          // } else if (remain == 2) {
          //   WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
          //   WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
          // } else if (remain == 1) {
          //   WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
          // }
          #ifndef NO_BIAS
            FLOAT4 dw2_outValue0 = RI_F(dw2_bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
          #else
            FLOAT4 dw2_outValue0 = (FLOAT4)(0.0f);
          #endif
          FLOAT4 dw2_outValue1 = dw2_outValue0;
          FLOAT4 dw2_outValue2 = dw2_outValue0;
          FLOAT4 dw2_outValue3 = dw2_outValue0;
          weights = RI_F(dw2_filter, SAMPLER, (int2)(0, inChannelBlockIdx));
          dw2_outValue0 = mad(outValue0, weights, dw2_outValue0);
          dw2_outValue1 = mad(outValue1, weights, dw2_outValue1);
          dw2_outValue2 = mad(outValue2, weights, dw2_outValue2);
          dw2_outValue3 = mad(outValue3, weights, dw2_outValue3);
          #ifdef DW2_RELU
            dw2_outValue0 = fmax(dw2_outValue0, (FLOAT4)0);
            dw2_outValue1 = fmax(dw2_outValue1, (FLOAT4)0);
            dw2_outValue2 = fmax(dw2_outValue2, (FLOAT4)0);
            dw2_outValue3 = fmax(dw2_outValue3, (FLOAT4)0);
          #endif
          #ifdef DW2_RELU6
            dw2_outValue0 = clamp(dw2_outValue0, (FLOAT4)0, (FLOAT4)6);
            dw2_outValue1 = clamp(dw2_outValue1, (FLOAT4)0, (FLOAT4)6);
            dw2_outValue2 = clamp(dw2_outValue2, (FLOAT4)0, (FLOAT4)6);
            dw2_outValue3 = clamp(dw2_outValue3, (FLOAT4)0, (FLOAT4)6);
          #endif
          #ifdef DW2_SIGMOID
            dw2_outValue0 = native_recip(1.0f + (FLOAT4)native_exp(-dw2_outValue0));
            dw2_outValue1 = native_recip(1.0f + (FLOAT4)native_exp(-dw2_outValue1));
            dw2_outValue2 = native_recip(1.0f + (FLOAT4)native_exp(-dw2_outValue2));
            dw2_outValue3 = native_recip(1.0f + (FLOAT4)native_exp(-dw2_outValue3));
          #endif
          // const int remain = dw1_outputShape.y - outWidthBlockidx4;
          // int outWidthIdx =
          //     mul24(outChannelBlockIdx, dw1_outputShape.y) + outWidthBlockidx4;
          // if (remain >= 4) {
          //   WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), dw2_outValue0);
          //   WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), dw2_outValue1);
          //   WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), dw2_outValue2);
          //   WI_F(output, (int2)(outWidthIdx + 3, outHeightBlockIdx), dw2_outValue3);
          // } else if (remain == 3) {
          //   WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), dw2_outValue0);
          //   WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), dw2_outValue1);
          //   WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), dw2_outValue2);
          // } else if (remain == 2) {
          //   WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), dw2_outValue0);
          //   WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), dw2_outValue1);
          // } else if (remain == 1) {
          //   WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), dw2_outValue0);
          // }
          #ifndef NO_BIAS
            FLOAT4 dw3_outValue0 = RI_F(dw3_bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
          #else
            FLOAT4 dw3_outValue0 = (FLOAT4)(0.0f);
          #endif
          FLOAT4 dw3_outValue1 = dw3_outValue0;
          FLOAT4 dw3_outValue2 = dw3_outValue0;
          FLOAT4 dw3_outValue3 = dw3_outValue0;
          weights = RI_F(dw3_filter, SAMPLER, (int2)(0, inChannelBlockIdx));
          dw3_outValue0 = mad(dw2_outValue0, weights, dw3_outValue0);
          dw3_outValue1 = mad(dw2_outValue1, weights, dw3_outValue1);
          dw3_outValue2 = mad(dw2_outValue2, weights, dw3_outValue2);
          dw3_outValue3 = mad(dw2_outValue3, weights, dw3_outValue3);
          #ifdef DW3_RELU
            dw3_outValue0 = fmax(dw3_outValue0, (FLOAT4)0);
            dw3_outValue1 = fmax(dw3_outValue1, (FLOAT4)0);
            dw3_outValue2 = fmax(dw3_outValue2, (FLOAT4)0);
            dw3_outValue3 = fmax(dw3_outValue3, (FLOAT4)0);
          #endif
          #ifdef DW3_RELU6
            dw3_outValue0 = clamp(dw3_outValue0, (FLOAT4)0, (FLOAT4)6);
            dw3_outValue1 = clamp(dw3_outValue1, (FLOAT4)0, (FLOAT4)6);
            dw3_outValue2 = clamp(dw3_outValue2, (FLOAT4)0, (FLOAT4)6);
            dw3_outValue3 = clamp(dw3_outValue3, (FLOAT4)0, (FLOAT4)6);
          #endif
          #ifdef DW3_SIGMOID
            dw3_outValue0 = native_recip(1.0f + (FLOAT4)native_exp(-dw3_outValue0));
            dw3_outValue1 = native_recip(1.0f + (FLOAT4)native_exp(-dw3_outValue1));
            dw3_outValue2 = native_recip(1.0f + (FLOAT4)native_exp(-dw3_outValue2));
            dw3_outValue3 = native_recip(1.0f + (FLOAT4)native_exp(-dw3_outValue3));
          #endif

          const int remain = dw1_outputShape.y - outWidthBlockidx4;
          int outWidthIdx =
              mul24(outChannelBlockIdx, dw1_outputShape.y) + outWidthBlockidx4;
          if (remain >= 4) {
            WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), dw3_outValue0);
            WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), dw3_outValue1);
            WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), dw3_outValue2);
            WI_F(output, (int2)(outWidthIdx + 3, outHeightBlockIdx), dw3_outValue3);
          } else if (remain == 3) {
            WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), dw3_outValue0);
            WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), dw3_outValue1);
            WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), dw3_outValue2);
          } else if (remain == 2) {
            WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), dw3_outValue0);
            WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), dw3_outValue1);
          } else if (remain == 1) {
            WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), dw3_outValue0);
          }
      }

__kernel
#if SET_ATTRIBUTE
    __attribute__((work_group_size_hint(16, 16, 1)))
#endif
    void
    resize1_s11k11_conv2_s22_add(
       GLOBAL_SIZE_3_DIMS __read_only image2d_t input1, __read_only image2d_t input2,
       __write_only image2d_t output, __private const float height_scale,
       __private const float width_scale, __private const float height_offset,
       __private const float width_offset, __private const int input_height,
       __private const int input_width, __private const int out_height
)
{
  const int outChannelBlockIdx = get_global_id(0);
  const int outWidthBlockIdx = get_global_id(1);
  const int outBatchHeightBlockIdx = get_global_id(2);
  DEAL_NON_UNIFORM_DIM3(outChannelBlockIdx, outWidthBlockIdx, outBatchHeightBlockIdx);
  // const int outChannelBlockIdxs = global_size_dim0;
  const int outWidth = global_size_dim1;
  const int outBatchIdx = outBatchHeightBlockIdx / out_height;
  const int outHeightIdx = outBatchHeightBlockIdx % out_height;
  const FLOAT scale_height = outHeightIdx * height_scale + height_offset;
  const FLOAT scale_width = outWidthBlockIdx * width_scale + width_offset;
  const int height_lf = max(0, (int)floor(scale_height));
  const int width_lf = max(0, (int)floor(scale_width));
  const int inWidthOffset = mul24(outChannelBlockIdx, input_width);
  const int inHeightOffset = mul24(outBatchIdx, input_height);
  FLOAT4 out1 = RI_F(
      input1, SAMPLER, (int2)(inWidthOffset + width_lf, inHeightOffset + height_lf));
  const int out_image_w =
      mad24(outChannelBlockIdx, outWidth, outWidthBlockIdx);
  const int out_image_h =
      mad24(outBatchIdx, out_height, outHeightIdx);
  FLOAT4 out2 = RI_F(input2, SAMPLER, (int2)(out_image_w, out_image_h));  
  out2 +=out1;
  WI_F(output, (int2)(out_image_w, out_image_h), out2);
}

__kernel
#if SET_ATTRIBUTE
    __attribute__((work_group_size_hint(16, 16, 1)))
#endif
    void
    conv1_s11k11_premap_add(GLOBAL_SIZE_2_DIMS 
                __read_only image2d_t conv1_input, __read_only image2d_t conv1_weights, __read_only image2d_t conv1_bias,
                __read_only image2d_t conv2_input, __write_only image2d_t output, __private const int2 inShape,
                __private const int in_channel_block,
                __private const int2 outShape,
                __private const int2 strideShape,
                __private const int outWidth4) 
{
  const int outChannelWidthIdx = get_global_id(0);
  const int outBatchHeightIdx = get_global_id(1);
  DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outBatchHeightIdx);
  const int outChannelBlockIdx = outChannelWidthIdx / outWidth4;
  const int outWidthBlockIdx = outChannelWidthIdx % outWidth4;
  FLOAT4 conv1_out0 = RI_F(conv1_bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
  FLOAT4 conv1_out1 = conv1_out0;
  FLOAT4 conv1_out2 = conv1_out0;
  FLOAT4 conv1_out3 = conv1_out0;

  int inWidthIdx0 = mul24(outWidthBlockIdx, strideShape.y * 4);
  int inWidthIdx1 = inWidthIdx0 + strideShape.y;
  int inWidthIdx2 = inWidthIdx1 + strideShape.y;
  int inWidthIdx3 = inWidthIdx2 + strideShape.y;
  inWidthIdx0 = select(inWidthIdx0, INT_MIN, inWidthIdx0 >= inShape.y);
  inWidthIdx1 = select(inWidthIdx1, INT_MIN, inWidthIdx1 >= inShape.y);
  inWidthIdx2 = select(inWidthIdx2, INT_MIN, inWidthIdx2 >= inShape.y);
  inWidthIdx3 = select(inWidthIdx3, INT_MIN, inWidthIdx3 >= inShape.y);
  int batchIndex = outBatchHeightIdx / outShape.x;
  int inHeightBlockIdx =
      mul24((outBatchHeightIdx % outShape.x), strideShape.x) +
      batchIndex * inShape.x;
  FLOAT4 conv1_in0, conv1_in1, conv1_in2, conv1_in3;
  FLOAT4 conv1_weights0, conv1_weights1, conv1_weights2, conv1_weights3;
  for (int inChannelBlockIdx = 0; inChannelBlockIdx < in_channel_block;
       ++inChannelBlockIdx) {
    int inWidthBase = inChannelBlockIdx * inShape.y;
    int weights_width_base = inChannelBlockIdx << 2;
    conv1_in0 = RI_F(
        conv1_input, SAMPLER, (int2)(inWidthBase + inWidthIdx0, inHeightBlockIdx));
    conv1_in1 = RI_F(
        conv1_input, SAMPLER, (int2)(inWidthBase + inWidthIdx1, inHeightBlockIdx));
    conv1_in2 = RI_F(
        conv1_input, SAMPLER, (int2)(inWidthBase + inWidthIdx2, inHeightBlockIdx));
    conv1_in3 = RI_F(
        conv1_input, SAMPLER, (int2)(inWidthBase + inWidthIdx3, inHeightBlockIdx));

    conv1_weights0 = RI_F(conv1_weights, SAMPLER,
                    (int2)(weights_width_base + 0, outChannelBlockIdx));
    conv1_weights1 = RI_F(conv1_weights, SAMPLER,
                    (int2)(weights_width_base + 1, outChannelBlockIdx));
    conv1_weights2 = RI_F(conv1_weights, SAMPLER,
                    (int2)(weights_width_base + 2, outChannelBlockIdx));
    conv1_weights3 = RI_F(conv1_weights, SAMPLER,
                    (int2)(weights_width_base + 3, outChannelBlockIdx));

    conv1_out0 = mad(conv1_in0.x, conv1_weights0, conv1_out0);                      
    conv1_out0 = mad(conv1_in0.y, conv1_weights1, conv1_out0);                      
    conv1_out0 = mad(conv1_in0.z, conv1_weights2, conv1_out0);                      
    conv1_out0 = mad(conv1_in0.w, conv1_weights3, conv1_out0);
    //-------------------------------------------------
    conv1_out1 = mad(conv1_in1.x, conv1_weights0, conv1_out1);                      
    conv1_out1 = mad(conv1_in1.y, conv1_weights1, conv1_out1);                      
    conv1_out1 = mad(conv1_in1.z, conv1_weights2, conv1_out1);                      
    conv1_out1 = mad(conv1_in1.w, conv1_weights3, conv1_out1);
    //-------------------------------------------------
    conv1_out2 = mad(conv1_in2.x, conv1_weights0, conv1_out2);                      
    conv1_out2 = mad(conv1_in2.y, conv1_weights1, conv1_out2);                      
    conv1_out2 = mad(conv1_in2.z, conv1_weights2, conv1_out2);                      
    conv1_out2 = mad(conv1_in2.w, conv1_weights3, conv1_out2);
    //--------------------------------------------------
    conv1_out3 = mad(conv1_in3.x, conv1_weights0, conv1_out3);                      
    conv1_out3 = mad(conv1_in3.y, conv1_weights1, conv1_out3);                      
    conv1_out3 = mad(conv1_in3.z, conv1_weights2, conv1_out3);                      
    conv1_out3 = mad(conv1_in3.w, conv1_weights3, conv1_out3);

  }

#ifdef CONV1_RELU
  conv1_out0 = fmax(conv1_out0, (FLOAT4)0);
  conv1_out1 = fmax(conv1_out1, (FLOAT4)0);
  conv1_out2 = fmax(conv1_out2, (FLOAT4)0);
  conv1_out3 = fmax(conv1_out3, (FLOAT4)0);
#endif
#ifdef CONV1_RELU6
  conv1_out0 = clamp(conv1_out0, (FLOAT4)0, (FLOAT4)6);
  conv1_out1 = clamp(conv1_out1, (FLOAT4)0, (FLOAT4)6);
  conv1_out2 = clamp(conv1_out2, (FLOAT4)0, (FLOAT4)6);
  conv1_out3 = clamp(conv1_out3, (FLOAT4)0, (FLOAT4)6);
#endif
// =========================================================
#ifdef CONV2_RELU
  conv2_out0 = fmax(conv2_out0, (FLOAT4)0);
  conv2_out1 = fmax(conv2_out1, (FLOAT4)0);
  conv2_out2 = fmax(conv2_out2, (FLOAT4)0);
  conv2_out3 = fmax(conv2_out3, (FLOAT4)0);
#endif
#ifdef CONV1_RELU6
  conv2_out0 = clamp(conv2_out0, (FLOAT4)0, (FLOAT4)6);
  conv2_out1 = clamp(conv2_out1, (FLOAT4)0, (FLOAT4)6);
  conv2_out2 = clamp(conv2_out2, (FLOAT4)0, (FLOAT4)6);
  conv2_out3 = clamp(conv2_out3, (FLOAT4)0, (FLOAT4)6);
#endif

  const int out_x_base = mul24(outChannelBlockIdx, outShape.y);
  int out_x_idx = outWidthBlockIdx << 2;
  const int remain = outShape.y - out_x_idx;
  int output_idx = out_x_base + out_x_idx;
  FLOAT4 conv2_out0, conv2_out1, conv2_out2, conv2_out3;
  if (remain >= 4) {
    conv2_out0 = RI_F(conv2_input, SAMPLER,
                    (int2)(output_idx, outBatchHeightIdx));
    conv1_out0 +=conv2_out0;

    conv2_out1 = RI_F(conv2_input, SAMPLER,
                    (int2)(output_idx + 1, outBatchHeightIdx));
    conv1_out1 +=conv2_out1;

    conv2_out2 = RI_F(conv2_input, SAMPLER,
                    (int2)(output_idx + 2, outBatchHeightIdx));
    conv1_out2 +=conv2_out2;

    conv2_out3 = RI_F(conv2_input, SAMPLER,
                    (int2)(output_idx + 3, outBatchHeightIdx));
    conv1_out3 +=conv2_out3;

    WI_F(output, (int2)(output_idx, outBatchHeightIdx), conv1_out0);
    WI_F(output, (int2)(output_idx + 1, outBatchHeightIdx), conv1_out1);
    WI_F(output, (int2)(output_idx + 2, outBatchHeightIdx), conv1_out2);
    WI_F(output, (int2)(output_idx + 3, outBatchHeightIdx), conv1_out3);

  } else if (remain == 3) {
    conv2_out0 = RI_F(conv2_input, SAMPLER,
                    (int2)(output_idx, outBatchHeightIdx));
    conv1_out0 +=conv2_out0;
    
    conv2_out1 = RI_F(conv2_input, SAMPLER,
                    (int2)(output_idx + 1, outBatchHeightIdx));
    conv1_out1 +=conv2_out1;

    conv2_out2 = RI_F(conv2_input, SAMPLER,
                    (int2)(output_idx + 2, outBatchHeightIdx));
    conv1_out2 +=conv2_out2;

    WI_F(output, (int2)(output_idx, outBatchHeightIdx), conv1_out0);
    WI_F(output, (int2)(output_idx + 1, outBatchHeightIdx), conv1_out1);
    WI_F(output, (int2)(output_idx + 2, outBatchHeightIdx), conv1_out2);

  } else if (remain == 2) {
    conv2_out0 = RI_F(conv2_input, SAMPLER,
                    (int2)(output_idx, outBatchHeightIdx));
    conv1_out0 +=conv2_out0;
    
    conv2_out1 = RI_F(conv2_input, SAMPLER,
                    (int2)(output_idx + 1, outBatchHeightIdx));
    conv1_out1 +=conv2_out1;

    WI_F(output, (int2)(output_idx, outBatchHeightIdx), conv1_out0);
    WI_F(output, (int2)(output_idx + 1, outBatchHeightIdx), conv1_out1);
    
  } else if (remain == 1) {
    conv2_out0 = RI_F(conv2_input, SAMPLER,
                    (int2)(output_idx, outBatchHeightIdx));
    conv1_out0 +=conv2_out0;

    WI_F(output, (int2)(output_idx, outBatchHeightIdx), conv1_out0);
  } 
}
__kernel
#if SET_ATTRIBUTE
    __attribute__((work_group_size_hint(16, 16, 1)))
#endif
    void
    dw1_s11k33_premap_add(
        GLOBAL_SIZE_2_DIMS 
        __read_only image2d_t input1,
        __read_only image2d_t input2,
        __write_only image2d_t output,
        __read_only image2d_t dw1_filter,
#ifndef NO_BIAS
        __read_only image2d_t dw1_bias,
#endif
        __private const int2 dw1_inputShape,
        __private const int dw1_inChannelBlocks,
        __private const int2 dw1_outputShape,
        __private const int2 dw1_filterShape,
        __private const int2 dw1_paddingShape
)
{
  const int outChannelWidthIdx = get_global_id(0);
  const int outHeightBlockIdx = get_global_id(1);
  DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightBlockIdx);
  int ow4 = (dw1_outputShape.y + 3) / 4;
  const int outChannelBlockIdx = outChannelWidthIdx / ow4;
  const int outWidthBlockidx = outChannelWidthIdx % ow4;
  const int inChannelBlockIdx = outChannelBlockIdx;
#ifndef NO_BIAS
  FLOAT4 outValue0 = RI_F(dw1_bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
#else
  FLOAT4 outValue0 = (FLOAT4)(0.0f);
#endif
  FLOAT4 outValue1 = outValue0;
  FLOAT4 outValue2 = outValue0;
  FLOAT4 outValue3 = outValue0;
  const int outWidthBlockidx4 = outWidthBlockidx << 2;
  const int inWidthOffset0 = outWidthBlockidx4 - dw1_paddingShape.y;
  const int inWidthOffset1 = inWidthOffset0 + 1;
  const int inWidthOffset2 = inWidthOffset0 + 2;
  const int inWidthOffset3 = inWidthOffset0 + 3;
  int heightIdx = outHeightBlockIdx % dw1_outputShape.x - dw1_paddingShape.x;
  const int outBatchIdx =
      mul24((outHeightBlockIdx / dw1_outputShape.x), dw1_inputShape.x);
  const int inCurIdx = mul24(inChannelBlockIdx, dw1_inputShape.y);

  const int inWidthIdx0 =
      select(inCurIdx + inWidthOffset0, -1,
             (inWidthOffset0 < 0 || inWidthOffset0 >= dw1_inputShape.y));
  const int inWidthIdx1 =
      select(inCurIdx + inWidthOffset1, -1,
             (inWidthOffset1 < 0 || inWidthOffset1 >= dw1_inputShape.y));
  const int inWidthIdx2 =
      select(inCurIdx + inWidthOffset2, -1,
             (inWidthOffset2 < 0 || inWidthOffset2 >= dw1_inputShape.y));

  FLOAT4 inValue0, inValue1, inValue2, inValue3;
  for (int kh = 0; kh < dw1_filterShape.x; kh++) {
    int inHeightIdx = select(heightIdx + outBatchIdx, -1,
                             (heightIdx < 0 || heightIdx >= dw1_inputShape.x));
    heightIdx++;
    inValue1 = RI_F(input1, SAMPLER, (int2)(inWidthIdx0, inHeightIdx));
    inValue2 = RI_F(input1, SAMPLER, (int2)(inWidthIdx1, inHeightIdx));
    inValue3 = RI_F(input1, SAMPLER, (int2)(inWidthIdx2, inHeightIdx));
    for (int kw = 0; kw < dw1_filterShape.y; kw++) {
      int filterIdx = mad24(kh, dw1_filterShape.y, kw);
      inValue0 = inValue1;
      inValue1 = inValue2;
      inValue2 = inValue3;

      int inWidthIdx = inWidthOffset3 + kw;
      inWidthIdx = select(inCurIdx + inWidthIdx, -1,
                          (inWidthIdx < 0 || inWidthIdx >= dw1_inputShape.y));
      inValue3 = RI_F(input1, SAMPLER, (int2)(inWidthIdx, inHeightIdx));
      FLOAT4 weights =
          RI_F(dw1_filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));
      outValue0 = mad(inValue0, weights, outValue0);
      outValue1 = mad(inValue1, weights, outValue1);
      outValue2 = mad(inValue2, weights, outValue2);
      outValue3 = mad(inValue3, weights, outValue3);
    }
  }
#ifdef DW1_RELU
  outValue0 = fmax(outValue0, (FLOAT4)0);
  outValue1 = fmax(outValue1, (FLOAT4)0);
  outValue2 = fmax(outValue2, (FLOAT4)0);
  outValue3 = fmax(outValue3, (FLOAT4)0);
#endif
#ifdef DW1_RELU6
  outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
  outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
  outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
  outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
#endif
  const int remain = dw1_outputShape.y - outWidthBlockidx4;
  int outWidthIdx0 =
      mul24(outChannelBlockIdx, dw1_outputShape.y) + outWidthBlockidx4;
  int outWidthIdx1 = outWidthIdx0 + 1, outWidthIdx2 = outWidthIdx0  + 2, outWidthIdx3 = outWidthIdx0 + 3;
  FLOAT4 dw2_outValue0 = 0, dw2_outValue1 = 0, dw2_outValue2 = 0, dw2_outValue3 = 0; 
  if (remain >= 4) {
    dw2_outValue0 = RI_F(input2, SAMPLER, (int2)(outWidthIdx0, outHeightBlockIdx));
    outValue0 += dw2_outValue0;
    dw2_outValue1 = RI_F(input2, SAMPLER, (int2)(outWidthIdx1, outHeightBlockIdx));
    outValue1 += dw2_outValue1;
    dw2_outValue2 = RI_F(input2, SAMPLER, (int2)(outWidthIdx2, outHeightBlockIdx));
    outValue2 += dw2_outValue2;
    dw2_outValue3 = RI_F(input2, SAMPLER, (int2)(outWidthIdx3, outHeightBlockIdx));
    outValue3 += dw2_outValue3;
    WI_F(output, (int2)(outWidthIdx0, outHeightBlockIdx), outValue0);
    WI_F(output, (int2)(outWidthIdx1, outHeightBlockIdx), outValue1);
    WI_F(output, (int2)(outWidthIdx2, outHeightBlockIdx), outValue2);
    WI_F(output, (int2)(outWidthIdx3, outHeightBlockIdx), outValue3);
  } else if (remain == 3) {
    dw2_outValue0 = RI_F(input2, SAMPLER, (int2)(outWidthIdx0, outHeightBlockIdx));
    outValue0 += dw2_outValue0;
    dw2_outValue1 = RI_F(input2, SAMPLER, (int2)(outWidthIdx1, outHeightBlockIdx));
    outValue1 += dw2_outValue1;
    dw2_outValue2 = RI_F(input2, SAMPLER, (int2)(outWidthIdx2, outHeightBlockIdx));
    outValue2 += dw2_outValue2;

    WI_F(output, (int2)(outWidthIdx0, outHeightBlockIdx), outValue0);
    WI_F(output, (int2)(outWidthIdx1, outHeightBlockIdx), outValue1);
    WI_F(output, (int2)(outWidthIdx2, outHeightBlockIdx), outValue2);
  } else if (remain == 2) {
    dw2_outValue0 = RI_F(input2, SAMPLER, (int2)(outWidthIdx0, outHeightBlockIdx));
    outValue0 += dw2_outValue0;
    dw2_outValue1 = RI_F(input2, SAMPLER, (int2)(outWidthIdx1, outHeightBlockIdx));
    outValue1 += dw2_outValue1;
    WI_F(output, (int2)(outWidthIdx0, outHeightBlockIdx), outValue0);
    WI_F(output, (int2)(outWidthIdx1, outHeightBlockIdx), outValue1);

  } else if (remain == 1) {
    dw2_outValue0 = RI_F(input2, SAMPLER, (int2)(outWidthIdx0, outHeightBlockIdx));
    outValue0 += dw2_outValue0;
    WI_F(output, (int2)(outWidthIdx0, outHeightBlockIdx), outValue0);
  }
}

__kernel
#if SET_ATTRIBUTE
    __attribute__((work_group_size_hint(16, 16, 1)))
#endif
    void
    dw1_s11k33_conv2_s11k11(
        GLOBAL_SIZE_2_DIMS 
        __private const int4 transChannelShape,
        __read_only image2d_t input,
        __write_only image2d_t output,
        __read_only image2d_t dw1_filter,
#ifndef NO_BIAS
        __read_only image2d_t dw1_bias,
#endif
        __private const int2 dw1_inputShape,
        __private const int dw1_inChannelBlocks,
        __private const int2 dw1_outputShape,
        __private const int2 dw1_filterShape,
        __private const int2 dw1_paddingShape,
        __read_only image2d_t conv2_filter,
#ifndef NO_BIAS
        __read_only image2d_t conv2_bias,
#endif  
        __private const int2 conv2_inputShape,
        __private const int2 conv2_outputShape,
        __private const int2 conv2_strideShape) {
          
        const int outWidthIdx = get_global_id(0);
        const int outBatchHeightIdx = get_global_id(1);
        DEAL_NON_UNIFORM_DIM2(outWidthIdx, outBatchHeightIdx);
        int ow4 = (dw1_outputShape.y + 3) / 4;
        const int outWidthBlockidx = outWidthIdx % ow4;
        int inCurIdx;
        int dw1_numOutChannels = transChannelShape.x/4, conv2_numOutChannels = transChannelShape.y/4, conv3_numOutChannels = transChannelShape.z/4; 
        #ifdef DW1_TEMP8
            FLOAT4 dw1_tempWidthOffset0[8], dw1_tempWidthOffset1[8], dw1_tempWidthOffset2[8], dw1_tempWidthOffset3[8]; 
        #endif
        #ifdef DW1_TEMP12
            FLOAT4 dw1_tempWidthOffset0[12], dw1_tempWidthOffset1[12], dw1_tempWidthOffset2[12], dw1_tempWidthOffset3[12]; 
        #endif
        #ifdef DW1_TEMP16
            FLOAT4 dw1_tempWidthOffset0[16], dw1_tempWidthOffset1[16], dw1_tempWidthOffset2[16], dw1_tempWidthOffset3[16]; 
        #endif
        #ifdef DW1_TEMP20
            FLOAT4 dw1_tempWidthOffset0[20], dw1_tempWidthOffset1[20], dw1_tempWidthOffset2[20], dw1_tempWidthOffset3[20]; 
        #endif
        #ifdef DW1_TEMP24
            FLOAT4 dw1_tempWidthOffset0[24], dw1_tempWidthOffset1[24], dw1_tempWidthOffset2[24], dw1_tempWidthOffset3[24]; 
        #endif
        #ifdef DW1_TEMP28
            FLOAT4 dw1_tempWidthOffset0[28], dw1_tempWidthOffset1[28], dw1_tempWidthOffset2[28], dw1_tempWidthOffset3[28]; 
        #endif
        #ifdef DW1_TEMP32
            FLOAT4 dw1_tempWidthOffset0[32], dw1_tempWidthOffset1[32], dw1_tempWidthOffset2[32], dw1_tempWidthOffset3[32]; 
        #endif
        #ifdef DW1_TEMP48
            FLOAT4 dw1_tempWidthOffset0[48], dw1_tempWidthOffset1[48], dw1_tempWidthOffset2[48], dw1_tempWidthOffset3[48]; 
        #endif
        #ifdef DW1_TEMP64
            FLOAT4 dw1_tempWidthOffset0[64], dw1_tempWidthOffset1[64], dw1_tempWidthOffset2[64], dw1_tempWidthOffset3[64]; 
        #endif
        #ifdef DW1_TEMP96
            FLOAT4 dw1_tempWidthOffset0[96], dw1_tempWidthOffset1[96], dw1_tempWidthOffset2[96], dw1_tempWidthOffset3[96]; 
        #endif
        #ifdef DW1_TEMP240
            FLOAT4 dw1_tempWidthOffset0[240], dw1_tempWidthOffset1[240], dw1_tempWidthOffset2[240], dw1_tempWidthOffset3[240]; 
        #endif
        
        FLOAT4 inValue0, inValue1, inValue2, inValue3;
        FLOAT4 outValue0, outValue1, outValue2, outValue3; 
        FLOAT4 weights, weights0, weights1, weights2, weights3;
        int outChannelBlockIdx, inChannelBlockIdx;
        int outWidthBlockidx4 = outWidthBlockidx << 2;
        int inWidthOffset0 = outWidthBlockidx4 - dw1_paddingShape.y;
        int inWidthOffset1 = inWidthOffset0 + 1;
        int inWidthOffset2 = inWidthOffset0 + 2;
        int inWidthOffset3 = inWidthOffset0 + 3;
        const int outBatchIdx = mul24((outBatchHeightIdx / dw1_outputShape.x), dw1_inputShape.x);
        // Start depthwise 1x1
        for(int outChannelBlockIdx = 0;outChannelBlockIdx < dw1_numOutChannels; outChannelBlockIdx++)
        {
          int heightIdx = outBatchHeightIdx % dw1_outputShape.x - dw1_paddingShape.x;
          inChannelBlockIdx = outChannelBlockIdx;
          outValue0 = RI_F(dw1_bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
          outValue1 = outValue0;
          outValue2 = outValue0;
          outValue3 = outValue0;
          inCurIdx = mul24(inChannelBlockIdx, dw1_inputShape.y);
          const int inWidthIdx0 =
            select(inCurIdx + inWidthOffset0, -1,
                  (inWidthOffset0 < 0 || inWidthOffset0 >= dw1_inputShape.y));
          const int inWidthIdx1 =
            select(inCurIdx + inWidthOffset1, -1,
                    (inWidthOffset1 < 0 || inWidthOffset1 >= dw1_inputShape.y));
          const int inWidthIdx2 =
            select(inCurIdx + inWidthOffset2, -1,
                    (inWidthOffset2 < 0 || inWidthOffset2 >= dw1_inputShape.y));
          for (int kh = 0; kh < dw1_filterShape.x; kh++) {
            int inHeightIdx = select(heightIdx + outBatchIdx, -1,
                                    (heightIdx < 0 || heightIdx >= dw1_inputShape.x));
            heightIdx++;
            inValue1 = RI_F(input, SAMPLER, (int2)(inWidthIdx0, inHeightIdx));
            inValue2 = RI_F(input, SAMPLER, (int2)(inWidthIdx1, inHeightIdx));
            inValue3 = RI_F(input, SAMPLER, (int2)(inWidthIdx2, inHeightIdx)); 
            for (int kw = 0; kw < dw1_filterShape.y; kw++) {
              int filterIdx = mad24(kh, dw1_filterShape.y, kw);
              inValue0 = inValue1;
              inValue1 = inValue2;
              inValue2 = inValue3;
              int inWidthIdx = inWidthOffset3 + kw;
              inWidthIdx = select(inCurIdx + inWidthIdx, -1,
                                  (inWidthIdx < 0 || inWidthIdx >= dw1_inputShape.y));
              inValue3 = RI_F(input, SAMPLER, (int2)(inWidthIdx, inHeightIdx));
              weights =
                  RI_F(dw1_filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));
              outValue0 = mad(inValue0, weights, outValue0);
              outValue1 = mad(inValue1, weights, outValue1);
              outValue2 = mad(inValue2, weights, outValue2);
              outValue3 = mad(inValue3, weights, outValue3);
            }
          }
          #ifdef DW1_RELU
            outValue0 = fmax(outValue0, (FLOAT4)0);
            outValue1 = fmax(outValue1, (FLOAT4)0);
            outValue2 = fmax(outValue2, (FLOAT4)0);
            outValue3 = fmax(outValue3, (FLOAT4)0);
          #endif
          #ifdef DW1_RELU6
            outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
            outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
            outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
            outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
          #endif
          #ifdef DW1_SIGMOID
            outValue0 = native_recip(1.0f + (FLOAT4)native_exp(-outValue0));
            outValue1 = native_recip(1.0f + (FLOAT4)native_exp(-outValue1));
            outValue2 = native_recip(1.0f + (FLOAT4)native_exp(-outValue2));
            outValue3 = native_recip(1.0f + (FLOAT4)native_exp(-outValue3));
          #endif

          dw1_tempWidthOffset0[outChannelBlockIdx] = outValue0;
          dw1_tempWidthOffset1[outChannelBlockIdx] = outValue1;
          dw1_tempWidthOffset2[outChannelBlockIdx] = outValue2;
          dw1_tempWidthOffset3[outChannelBlockIdx] = outValue3;
        }
        // Start convolution 1x1
        int conv2_batchIndex = outBatchHeightIdx / conv2_outputShape.x;
        // int input_height_block_idx =
        //     mul24((output_batch_height_idx % output_shape.x), stride_shape.x) +
        //     batch_index * input_shape.x;
        int conv2_numInChannels = transChannelShape.x / 4;
        for(int conv2_outChannelBlockIdx = 0;conv2_outChannelBlockIdx < conv2_numOutChannels; conv2_outChannelBlockIdx++)
        {
          outValue0 = RI_F(conv2_bias, SAMPLER, (int2)(conv2_outChannelBlockIdx, 0));
          outValue1 = outValue0;
          outValue2 = outValue0;
          outValue3 = outValue0;
          // in##i
          for(int conv2_inChannelBlockidx = 0; conv2_inChannelBlockidx < conv2_numInChannels;
              ++conv2_inChannelBlockidx)
          {
            int conv2_inWidthBase = conv2_inChannelBlockidx * conv2_inputShape.y;
            int conv2_FilterWidthBase = conv2_inChannelBlockidx << 2;
            weights0 = RI_F(conv2_filter, SAMPLER, 
                            (int2)(conv2_FilterWidthBase + 0, conv2_outChannelBlockIdx));
            weights1 = RI_F(conv2_filter, SAMPLER, 
                            (int2)(conv2_FilterWidthBase + 1, conv2_outChannelBlockIdx));
            weights2 = RI_F(conv2_filter, SAMPLER, 
                            (int2)(conv2_FilterWidthBase + 2, conv2_outChannelBlockIdx));
            weights3 = RI_F(conv2_filter, SAMPLER, 
                            (int2)(conv2_FilterWidthBase + 3, conv2_outChannelBlockIdx));
            inValue0 = dw1_tempWidthOffset0[conv2_inChannelBlockidx]; 
            inValue1 = dw1_tempWidthOffset1[conv2_inChannelBlockidx];
            inValue2 = dw1_tempWidthOffset2[conv2_inChannelBlockidx];
            inValue3 = dw1_tempWidthOffset3[conv2_inChannelBlockidx];
            CALCULATE_OUTPUT(0);
            CALCULATE_OUTPUT(1);
            CALCULATE_OUTPUT(2);
            CALCULATE_OUTPUT(3);
          }
          #ifdef CONV2_RELU
            outValue0 = fmax(outValue0, (FLOAT4)0);
            outValue1 = fmax(outValue1, (FLOAT4)0);
            outValue2 = fmax(outValue2, (FLOAT4)0);
            outValue3 = fmax(outValue3, (FLOAT4)0);
          #endif
          #ifdef CONV2_RELU6
            outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
            outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
            outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
            outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
          #endif
          #ifdef CONV2_SIGMOID
            outValue0 = native_recip(1.0f + (FLOAT4)native_exp(-outValue0));
            outValue1 = native_recip(1.0f + (FLOAT4)native_exp(-outValue1));
            outValue2 = native_recip(1.0f + (FLOAT4)native_exp(-outValue2));
            outValue3 = native_recip(1.0f + (FLOAT4)native_exp(-outValue3));
          #endif
          const int out_x_base = mul24(conv2_outChannelBlockIdx, conv2_outputShape.y);
          int out_x_idx = outWidthBlockidx << 2;
          const int remain = conv2_outputShape.y - out_x_idx;
          int output_idx = out_x_base + out_x_idx;
          if(remain >= 4)
          { 
            WI_F(output, (int2)(output_idx, outBatchHeightIdx), outValue0);
            WI_F(output, (int2)(output_idx + 1, outBatchHeightIdx), outValue1);
            WI_F(output, (int2)(output_idx + 2, outBatchHeightIdx), outValue2);
            WI_F(output, (int2)(output_idx + 3, outBatchHeightIdx), outValue3);
          }else if(remain == 3)
          {
            WI_F(output, (int2)(output_idx, outBatchHeightIdx), outValue0);
            WI_F(output, (int2)(output_idx + 1, outBatchHeightIdx), outValue1);
            WI_F(output, (int2)(output_idx + 2, outBatchHeightIdx), outValue2);
          }else if(remain == 2)
          {
            WI_F(output, (int2)(output_idx, outBatchHeightIdx), outValue0);
            WI_F(output, (int2)(output_idx + 1, outBatchHeightIdx), outValue1);
          }else if(remain == 1)
          {
            WI_F(output, (int2)(output_idx, outBatchHeightIdx), outValue0);
          }
        }

}