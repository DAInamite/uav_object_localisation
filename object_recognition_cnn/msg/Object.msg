Header header

float64 position_x
float64 position_y
uint8 object_type # 0 means base, 1 means battery, 2 means pot
int32 imgWidth
int32 imgHeight
float64 object_probability # The probability that this object is of the type saved in "object_type"
float64 position_probability # The probability that this object is at the position(position_x, position_y)