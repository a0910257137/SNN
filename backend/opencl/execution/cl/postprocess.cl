#define GLOBAL_SIZE_2_DIMS                                                     \
  __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                  \
  if (input1 >= global_size_dim0 || input2 >= global_size_dim1) {              \
    return;                                                                    \
  }
#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

__constant sampler_t SAMPLER =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
// only for debug
// convert data from image(b h, ic/4 w ic4) to buffer(nhwc)
__kernel void genBbboxLandmarks(GLOBAL_SIZE_2_DIMS
#ifdef BUFFER_IMAGE_IO_TRANS
                                __global float *bbox_output,
                                __global float *params_output,
                                __global float *trans_output,
#else
                                __global FLOAT *bbox_output,
                                __global FLOAT *params_output,  
                                __global FLOAT *trans_output, 
#endif
                                __private const int height,
                                __private const int width,
                                __private const int cls_channels,
                                __private const int bbox_channels,
                                __private const int params_channels,
                                __private  int trans_channels,
                                __read_only image2d_t cls_input_ptr,
                                __read_only image2d_t bbox_input_ptr,
                                __read_only image2d_t params_input_ptr,
                                __read_only image2d_t trans_input_ptr){
    int imageWidthIdx = get_global_id(0);
    int imageHeightIdx = get_global_id(1);
    // int imageChannelIdx = get_global_id(2);
    DEAL_NON_UNIFORM_DIM2(imageWidthIdx, imageHeightIdx);
    const int batchIdx = imageHeightIdx / height;
    const int heightIdx = imageHeightIdx % height;
    const int widthIdx = imageWidthIdx % width;
    const int channel_4_idx = (imageWidthIdx / width) << 2;
    const int buffer_offset =
        ((batchIdx * height + heightIdx) * width + widthIdx) * cls_channels +
        channel_4_idx;
    int2 coord = (int2)(imageWidthIdx, imageHeightIdx);
    // printf("%d\n", coord.x);
    #ifdef BUFFER_IMAGE_IO_TRANS
        float4 cls_values = convert_float4(RI_F(cls_input_ptr, SAMPLER, coord));
    #else   
        FLOAT4 cls_values = RI_F(cls_input_ptr, SAMPLER, coord);
    #endif
    FLOAT4 cls_scores = native_recip(1.0f + (FLOAT4)native_exp(-cls_values));
    int yidx, xidx, cidx, start, end;
    int bboxBlk = bbox_channels / cls_channels;
    int paramBlk = params_channels / cls_channels;
    int transBlk = trans_channels / cls_channels;
    float4 clsValues, bboxValues, paramValues, transValues;
    // cls_channels = 4
    const int remain_channel = cls_channels - channel_4_idx;
    bool isPeak = false;
    yidx = heightIdx;
    xidx = widthIdx;
    int offset = yidx * (width * cls_channels) + xidx * cls_channels;
    int cls_offset = buffer_offset;
    if (remain_channel >= 4)
    {
        if(cls_scores.x > 0.5){ 
            isPeak = true;
        }else if (cls_scores.y > 0.5){
            isPeak = true;
            cls_offset += 1;
        } else if (cls_scores.z > 0.5){
            isPeak = true;
            cls_offset += 2;
        } else if(cls_scores.w > 0.5){
            isPeak = true;
            cls_offset += 3;
        }
        if(isPeak)
        {
            int stride = 16;
            int stride_width = 20;
            int mesh_x = widthIdx * stride;
            int mesh_y = heightIdx * stride;
            // for parameters 
            cidx = (cls_offset - offset) % 2;
            start = cidx == 0 ? 0 : paramBlk/2;
            end = cidx == 0 ? paramBlk/2 : paramBlk;
            for(; start < end; ++start)
            {   
                int2 out_coord = (coord.x + start*width, coord.y);
                // could add pms calculate in here?
                paramValues = convert_float4(RI_F(params_input_ptr, SAMPLER, out_coord));
                vstore4(paramValues, 0, params_output + buffer_offset);
            }
            // for bboxes 
            start = cidx == 0 ? 0 : bboxBlk/2;
            end = cidx == 0 ? bboxBlk/2 : bboxBlk;
            for(; start < end; ++start)
            {   
                int2 out_coord = (coord.x + start*width, coord.y);
                bboxValues = convert_float4(RI_F(bbox_input_ptr, SAMPLER, out_coord));
                bboxValues *= (float) stride;
                bboxValues.x = MIN(MAX(0, mesh_x - bboxValues.x), stride_width * stride);
                bboxValues.y = MIN(MAX(0, mesh_y - bboxValues.y), stride_width * stride);
                bboxValues.z = MIN(MAX(0, mesh_x + bboxValues.z), stride_width * stride);
                bboxValues.w = MIN(MAX(0, mesh_y + bboxValues.w), stride_width * stride);
                vstore4(bboxValues, 0, bbox_output + buffer_offset);
            }
            // for trans
            transValues = convert_float4(RI_F(trans_input_ptr, SAMPLER, coord));
            start = cidx == 0 ? 0 : transBlk/2;
            end = cidx == 0 ? transBlk/2 : transBlk;
            int fillOffsetIdx = buffer_offset;
            if(cidx == 0)
            {
                transValues.x = MIN(MAX(0, mesh_x + transValues.x), stride_width * stride);
                transValues.y = MIN(MAX(0, mesh_y + transValues.y), stride_width * stride);
                trans_output[fillOffsetIdx] = transValues.x;
                trans_output[fillOffsetIdx + 1] = transValues.y;

            }
            else
            {
                transValues.z = MIN(MAX(0, mesh_x + transValues.z), stride_width * stride);
                transValues.w = MIN(MAX(0, mesh_y + transValues.w), stride_width * stride);
                trans_output[fillOffsetIdx + 2] = transValues.z;
                trans_output[fillOffsetIdx + 3] = transValues.w;
            }
            // vstore4(transValues, 0, trans_output + buffer_offset);
        }
    }
    // cls formula o_pos = j * (width * channel) + i * channel;
    
    
}