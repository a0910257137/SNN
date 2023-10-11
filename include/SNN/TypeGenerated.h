#ifndef TYPEGENERATED_H
#define TYPEGENERATED_H
enum DataFormat
{
    DATA_FORMAT_NHWC = 0,
    DATA_FORMAT_NCHW,
};
enum FilterFormat
{
    FILTER_FORMAT_OHWI = 0,
    FILTER_FORMAT_OIHW = 1,
    FILTER_FORMAT_IOHW = 2,
    FILTER_FORMAT_MHWI = 3,
    FILTER_FORMAT_MIHW = 4,

};
enum DataType
{
    DataType_DT_INVALID = 0,
    DataType_DT_FLOAT = 1,
    DataType_DT_DOUBLE = 2,
    DataType_DT_INT32 = 3,
    DataType_DT_UINT8 = 4,
    DataType_DT_INT16 = 5,
    DataType_DT_INT8 = 6,
    DataType_DT_STRING = 7,
    DataType_DT_COMPLEX64 = 8,
    DataType_DT_INT64 = 9,
    DataType_DT_BOOL = 10,
    DataType_DT_QINT8 = 11,
    DataType_DT_QUINT8 = 12,
    DataType_DT_QINT32 = 13,
    DataType_DT_BFLOAT16 = 14,
    DataType_DT_QINT16 = 15,
    DataType_DT_QUINT16 = 16,
    DataType_DT_UINT16 = 17,
    DataType_DT_COMPLEX128 = 18,
    DataType_DT_HALF = 19,
    DataType_DT_RESOURCE = 20,
    DataType_DT_VARIANT = 21,
    DataType_MIN = DataType_DT_INVALID,
    DataType_MAX = DataType_DT_VARIANT
};

inline const DataType (&EnumValuesDataType())[22]
{
    static const DataType values[] = {
        DataType_DT_INVALID,
        DataType_DT_FLOAT,
        DataType_DT_DOUBLE,
        DataType_DT_INT32,
        DataType_DT_UINT8,
        DataType_DT_INT16,
        DataType_DT_INT8,
        DataType_DT_STRING,
        DataType_DT_COMPLEX64,
        DataType_DT_INT64,
        DataType_DT_BOOL,
        DataType_DT_QINT8,
        DataType_DT_QUINT8,
        DataType_DT_QINT32,
        DataType_DT_BFLOAT16,
        DataType_DT_QINT16,
        DataType_DT_QUINT16,
        DataType_DT_UINT16,
        DataType_DT_COMPLEX128,
        DataType_DT_HALF,
        DataType_DT_RESOURCE,
        DataType_DT_VARIANT};
    return values;
}

inline const char *const *EnumNamesDataType()
{
    static const char *const names[] = {
        "DT_INVALID",
        "DT_FLOAT",
        "DT_DOUBLE",
        "DT_INT32",
        "DT_UINT8",
        "DT_INT16",
        "DT_INT8",
        "DT_STRING",
        "DT_COMPLEX64",
        "DT_INT64",
        "DT_BOOL",
        "DT_QINT8",
        "DT_QUINT8",
        "DT_QINT32",
        "DT_BFLOAT16",
        "DT_QINT16",
        "DT_QUINT16",
        "DT_UINT16",
        "DT_COMPLEX128",
        "DT_HALF",
        "DT_RESOURCE",
        "DT_VARIANT",
        nullptr};
    return names;
}
#endif