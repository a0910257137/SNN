#define GLOBAL_SIZE_2_DIMS                                                     \
  __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                  \
  if (input1 >= global_size_dim0 || input2 >= global_size_dim1) {              \
    return;                                                                    \
  }

// convert kernel from buffer(mihw) to image(ic/4, ic4 h w m)
// but now dw only support m == 1
__kernel void dw_filter_buffer_to_image(GLOBAL_SIZE_2_DIMS
#ifdef BUFFER_INP_FP32
                                            __global const float *input_ptr,
#else
                                            __global const FLOAT *input_ptr,
#endif
                                        __private const int4 kernel_shape,
                                        __private const int height_width_size,
                                        __write_only image2d_t output) {

  const int image_width_idx = get_global_id(0);
  const int image_height_idx = get_global_id(1);
  DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);
  FLOAT4 output_values = 0;
  if (kernel_shape.x == 1) {
    const int input_channel_4_idx = image_height_idx * 4;
    const int buffer_height_idx = image_width_idx / kernel_shape.w;
    const int buffer_width_idx = image_width_idx % kernel_shape.w;
    const int buffer_offset =
        mad24(mad24(input_channel_4_idx, kernel_shape.z, buffer_height_idx),
              kernel_shape.w, buffer_width_idx);
    const int remain_channel = kernel_shape.y - input_channel_4_idx;
    if (input_channel_4_idx < kernel_shape.y) {
      if (remain_channel >= 4) {
        int offset = buffer_offset;
        output_values.x = (FLOAT)(*(input_ptr + offset));
        offset += height_width_size;
        output_values.y = (FLOAT)(*(input_ptr + offset));
        offset += height_width_size;
        output_values.z = (FLOAT)(*(input_ptr + offset));
        offset += height_width_size;
        output_values.w = (FLOAT)(*(input_ptr + offset));
      } else if (remain_channel == 3) {
        int offset = buffer_offset;
        output_values.x = (FLOAT)(*(input_ptr + offset));
        offset += height_width_size;
        output_values.y = (FLOAT)(*(input_ptr + offset));
        offset += height_width_size;
        output_values.z = (FLOAT)(*(input_ptr + offset));
      } else if (remain_channel == 2) {
        int offset = buffer_offset;
        output_values.x = (FLOAT)(*(input_ptr + offset));
        offset += height_width_size;
        output_values.y = (FLOAT)(*(input_ptr + offset));
      } else if (remain_channel == 1) {
        int offset = buffer_offset;
        output_values.x = (FLOAT)(*(input_ptr + offset));
      }
    }
  }
  WI_F(output, (int2)(image_width_idx, image_height_idx), output_values);
}