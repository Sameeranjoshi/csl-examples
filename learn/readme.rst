
/*
Types: 
    - int       i32/ ui32
    - float     f32
    - double    - missing
    - char      - missing
    - string    - missing
    - bool      bool
*/
/*
Functions:
    bool fn (){

    }

    or 
    bool @fadd(){}
    
    or

*/

/*
Control Structures:
    -  if statement
    - while and for loop
*/


# RUn commands

    cslc layout.csl --fabric-dims=8,8 --fabric-offsets=4,1 --memcpy --channels=1 -o out
    cs_python run.py --name out/

# Using cs_readelf
        cs_readelf out/bin/out_0_0.elf --visualize=fine

        # Check the basic details
        cs_readelf -s out/bin/out_0_0.elf

        # Look at fabric configuration
        cs_readelf --fs out/bin/out_0_0.elf
        cs_readelf --visualize=coarse out/bin/out_0_0.elf

        # Analyze symbols and memory usage
        cs_readelf --symbols out/bin/out_0_0.elf
        cs_readelf -m out/bin/out_0_0.elf

        # Dump specific sections if needed
        cs_readelf --xp=0 out/bin/out_0_0.elf
