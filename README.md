# WPS-ERA5-PREPROCESSOR
Python code and notebook for processing ERA5 reanalysis NetCDF files into the WPS intermediate format.
ERA5 files in NetCDF format are available from the NCAR RDA.
This represents a modification and extension of code written by Luke Madaus and published at
https://gitlab.com/jupiter-opensource/wrf-preproc-intermediate-writer

Robert Fovell, rfovell@albany.edu.  No promises or warranty or support.

The notebook WPS_ERA5_PREPROCESSOR.ipynb provides an example of using this code.

----------------------------------------------------------------------------------------------
BEFORE RUNNING NOTEBOOK WPS_ERA5_PREPROCESSOR.ipynb
----------------------------------------------------------------------------------------------

(0) Install the Python script file wps_formatter_ERA5_RGF.py somewhere that is accessible by the notebook.

(1) Download ERA5 files from NCAR RDA in NetCDF format.<br>


  Isobaric level fields needed: z, t, u, v, r.  (q is not needed) <br>
  Surface fields needed: msl, sp, 2t, sstk, skt, 10u, 10v, 2d <br>
  Subsurface fields needed: stl1, stl2, stl3, stl4, swvl1, swvl2, swvl3, swvl4 <br>
  
(2) For multiple day runs, combine isobaric level files by field.  This can take a lot of time.

ncrcat e5.oper.an.pl.128_129_z* era5_z.nc <br>
ncrcat e5.oper.an.pl.128_130_t* era5_t.nc <br>
ncrcat e5.oper.an.pl.128_131_u* era5_u.nc <br>
ncrcat e5.oper.an.pl.128_132_v* era5_v.nc <br>
ncrcat e5.oper.an.pl.128_157_r* era5_r.nc <br>

(3) The surface fields each encompass an entire month.  For runs crossing months, you may need to 
 combine surface files or run this notebook and do the following steps more than once

(4) <b>ERA5 does not supply 2 m RH</b>.  Not having 2 m RH can also cause real.exe to ignore U10 and V10 information,
 changing fields near the surface, because it forces use_surface to False 
    + Run notebook <b>ERA5_RH2m_computer.ipynb</b> to read in 2t, 2d files, computing 2m RH via MetPy <br>
    + This generates a new NetCDF file called output_file.nc <br>
    + Then change name of variable in output file:  ncrename -v VAR_2D,VAR_2RH output_file.nc <br>
    + Change name of output_file.nc if needed:      mv output_file.nc era5_rh2m_aug2019.nc <br>
    
(5) Have the invariant fields SOILHGT available and expressed in meters (not surface geopotential in m2/s2).
 The file era5_SOILHGT.nc is provided here in this repository.  The invariant field LANDSEA can be found in
 the file era5_LANDSEA.nc
 
(6) Modify Cell #2 for locations, names of files

(7) Also in Cell #2, point to location of the grib_codes.df.pkl file, which is provided in this repository.

(8) In Cell #3, set the time range.  Example: <br>
'time_range' : (datetime(2019,8,26,12), datetime(2019,8,27,0)),<br>

(9) In Cell #3, set the time interval (interval_seconds, expressed in hours).  Example:<br>
'time_freq' : '3h',<br>

(10) In the metgrid step, do NOT include the ERA5 invariant file that you would use with GRIB data.
This appears to corrupt the soils information.

The notebook creates a set of files named "FILE:" followed by a timestamp similar to what ungrib.exe does.

NOTE: The notebook ERA5_RH2m_computer.ipynb computes 2 m RH using MetPy, which produces slightly different results from 
what ungrib.exe creates (in rrpr.F).


NOTE added 2/24/2025: It was pointed out to me that ERA5's relative humidity (RH) is a weighted average of RH with
respect to liquid (RH_l) and ice (RH_i) for below-freezing temperatures and ungrib.exe adjusts for this because real.exe
expects RH_l.  This is not currently accounted for in this code.  Additionally, it appears ungrib.exe does not handle this 
in a manner that is consistent with ERA5's documentation, so even ungrib.exe is not correct.  It also does not appear that 
simply replacing RH with specific humidity (Q) will suffice as my tests using Q in place of RH in ungrib.exe (not this 
preprocessor) do not produce the same results.  Work in progress.
[end]
