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
FLOAT4 stitch_vector(FLOAT4 left, FLOAT4 right, const int pos,
                     const bool reversed) {
  if (!reversed) {
    switch (pos) {
    case 1:
      return (FLOAT4)(left.x, right.x, right.y, right.z);
    case 2:
      return (FLOAT4)(left.x, left.y, right.x, right.y);
    case 3:
      return (FLOAT4)(left.x, left.y, left.z, right.x);
    default:
      return (FLOAT4)0;
    }
  } else {
    switch (pos) {
    case 1:
      return (FLOAT4)(left.w, right.x, right.y, right.z);
    case 2:
      return (FLOAT4)(left.z, left.w, right.x, right.y);
    case 3:
      return (FLOAT4)(left.y, left.z, left.w, right.x);
    default:
      return (FLOAT4)0;
    }
  }
}
// Supported data type: half/float
__kernel void concat_channel(GLOBAL_SIZE_3_DIMS __read_only image2d_t input0,
                             __read_only image2d_t input1,
                             __private const int input0_chan,
                             __private const int input1_chan,
                             __write_only image2d_t output) {
  const int chan_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

  DEAL_NON_UNIFORM_DIM3(chan_blk_idx, width_idx, hb_idx);

  const int width = global_size_dim1;
  const int output_chan = input0_chan + input1_chan;
  const int input0_chan_blk = (input0_chan + 3) >> 2;
  const int output_chan_blk = (output_chan + 3) >> 2;

  FLOAT4 data = 0;
#ifdef DIVISIBLE_FOUR
  if (chan_blk_idx + 1 <= input0_chan_blk) {

    data = RI_F(input0, SAMPLER,
                (int2)(mad24(chan_blk_idx, width, width_idx), hb_idx));
  } else {

    data =
        RI_F(input1, SAMPLER,
             (int2)(mad24((chan_blk_idx - input0_chan_blk), width, width_idx),
                    hb_idx));
  }
#else
  if (chan_blk_idx < input0_chan_blk - 1) {
    data = RI_F(input0, SAMPLER,
                (int2)(mad24(chan_blk_idx, width, width_idx), hb_idx));
  } else if (chan_blk_idx >= input0_chan_blk) {

    const int in_chan_idx = chan_blk_idx - input0_chan_blk;
    // take input1 [input0, input1]
    // last item needs to fill in 0., because float4 x, y, w, z=0

    FLOAT4 data0 = RI_F(input1, SAMPLER,
                        (int2)(mad24(in_chan_idx, width, width_idx), hb_idx));
    FLOAT4 data1 = 0;

    if (((in_chan_idx + 1) << 2) < input1_chan) {
      data1 = RI_F(input1, SAMPLER,
                   (int2)(mad24((in_chan_idx + 1), width, width_idx), hb_idx));
    }

    data = stitch_vector(data0, data1, input0_chan % 4, true);
  } else { // if (chan_blk_idx == input0_chan_blk - 1)
    FLOAT4 data0 = RI_F(input0, SAMPLER,
                        (int2)(mad24(chan_blk_idx, width, width_idx), hb_idx));

    FLOAT4 data1 = RI_F(input1, SAMPLER, (int2)(width_idx, hb_idx));

    data = stitch_vector(data0, data1, input0_chan % 4, false);
  }
#endif

  const int pos = mad24(chan_blk_idx, width, width_idx);

  WI_F(output, (int2)(pos, hb_idx), data);
}

// // Required: All input channels are divisible by 4
// __kernel void
// concat_channel_multi(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
//                      __private const int chan_blk_offset,
//                      __write_only image2d_t output) {
//   const int chan_blk_idx = get_global_id(0);
//   const int width_idx = get_global_id(1);
//   const int hb_idx = get_global_id(2);

// #ifndef NON_UNIFORM_WORK_GROUP
//   if (chan_blk_idx >= global_size_dim0 || width_idx >= global_size_dim1 ||
//       hb_idx >= global_size_dim2) {
//     return;
//   }
// #endif
//   const int width = global_size_dim1;

//   FLOAT4 data = 0;
//   data = RI_F(input, SAMPLER,
//               (int2)(mad24(chan_blk_idx, width, width_idx), hb_idx));

//   const int pos = mad24(chan_blk_idx + chan_blk_offset, width, width_idx);

//   WI_F(output, (int2)(pos, hb_idx), data);
// }
