"""
#####################################################################
Intermediate file writer for ERA5 reanalysis fields in NetCDF format
Modified from wrf-preproc-intermediate-writer-master/wps_formatter.py
  written by Luke Madaus, published at https://gitlab.com/jupiter-opensource/wrf-preproc-intermediate-writer

Robert Fovell rfovell@albany.edu, no promises or warranty or support
Version of 12/5/2024

NOTES: (look for "RGF" in code)
(1) Geographic bounds are not working at this time
(2) I had to force the output levels to be in pascals
(3) I rewrote add_to_WPS to handle 2 m and 10 m fields that have same 
  metgrid names as isobaric level fields first
(4) For ERA5, which does not have 2 m RH, I compute this field first
  before preprocessing, using 2 m T and dewpoint in separate notebook
(5) ERA5's geopotential field is converted to geopotential height
(6) I added a lot of variable fields to Luke's grib_codes table
(7) The ERA5 surface geopotential is already converted to SOILHGT
 
#####################################################################
Copyright 2020 Jupiter Intelligence, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""
WPS intermediate formatter

Goal:  Take arbitrary arrays (as xarray DataArrays) 
and write them to WRF/WPS intermediate format for
further processing by WRF

Originator: Luke Madaus <luke.madaus@jupiterintel.com>
Jupiter
1320 Pearl St, Suite 232
Boulder, CO

Modified by: Robert Fovell (rfovell@albany.edu)

"""
from datetime import datetime, timedelta
import logging
import numpy as np
import xarray
import pandas as pd
import os
import sys
from netCDF4 import Dataset, num2date

# from netcdftime import utime
from struct import pack, calcsize
from scipy.io import FortranFile
from scipy.spatial import KDTree
from multiprocessing import cpu_count

# Set up default logger
# Master logging object
module_logger = logging.Logger("WRFInputFormatter")
# This stream handler will write to the console
console_logger = logging.StreamHandler()
# Set it so it prints info-level logs to console
console_logger.setLevel(logging.INFO)
# Set how these log messages should be formatted
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_logger.setFormatter(formatter)
# Add this stream to the master logger
module_logger.addHandler(console_logger)


# Hacky way of non-uniform variable name resolution
alternate_variable_names = {
    "ta": "t",
    "hus": "q",
    "ua": "u",
    "va": "v",
    "ps": "sp",
    "psl": "msl",
    "tas": "t2m",
    "uas": "u10",
    "vas": "v10",
    "ts": "skt",
}


class WRFInputFormatter:
    """
    Object that takes a set of possible fields
    for input to WRF and converts them to a WRF-usable format
    Requires:
    - time_range -> Tuple of datetime objects representing (start, end) of desired period
    - variable_table -> For now, a pandas dataframe of info about each variable type from this source
    - master_path -> Path to where source netCDF files are located
    - time_freq -> How frequently to produce forecast files within that time_range.  Default is '6H'
    - vcoord_type -> Type of vertical coordinate here.  Options are 'pressure', 'hybrid-z' and 'hybrid-p'
    - vcoord -> The name of the vertical coordinate for 3d fields
    - soilcoord -> The name of the soil coordinate for soil fields 
    - outfile_format -> Format for dumping the output file. Default (and only) option -> 'WPS'
    - plevels -> List or array of pressure levels (in Pa) to interpolate to if pressure level interpolation is required
    - projection -> Integer code for the way the data is projected in the WPS format output.  Default is 0
    - geog_bounds -> A dictionary of lat/lon values to use as a bounding box for subsetting the input data 
        (fmt: {'north': north_lat, 'south': south_lat, 'west': west_lon, 'east': east_lon}). Lons in 0-365.
    - use_variables -> A list of variables to use from this input dataset for WRF processing
    - ps_path -> Optional secondary path for surface pressure (PS); required to convert some model levels to pressure levels

    """

    def __init__(
        self,
        time_range,
        variable_table=None,
        master_path=None,
        variable_paths=None,
        time_freq="6H",
        vcoord_type="hybrid-z",
        vcoord="lev",
        soilcoord="level",
        outfile_format="WPS",
        plevels=[30000.0, 20000.0],
        projection=0,
        geog_bounds=None,
        use_variables=["ta", "ua", "va", "hus", "ps", "tsl", "mrlsl", "ts"],
        ps_path=None,
        infile_format="netCDF",
        dask_client=None,
        dask_cluster=None,
        time_coord="time",
        logger=None,
    ):
        # Set up the logger
        if logger is None:
            self.logger = module_logger
        else:
            self.logger = logger

        if variable_table is None:
            self.variable_table = pd.read_pickle("grib_codes.df.pkl")
        else:
            self.variable_table = variable_table
        if variable_paths is not None:
            self.variable_paths = variable_paths
        self.master_path = master_path
        self.time_range = time_range
        self.time_freq = time_freq
        self.timecoord = time_coord
        self.vcoord_type = vcoord_type
        self.vcoord = vcoord
        self.soilcoord = soilcoord
        self.outfile_format = outfile_format
        self.projection = projection
        self.geog_bounds = geog_bounds
        self.use_variables = use_variables
        self.ps_path = ps_path
        self.infile_format = infile_format

        self.dask_cluster = dask_cluster
        self.dask_client = dask_client
        self.terminate_dask_on_complete = False
        if self.infile_format in ["zarr"] and (self.dask_client is None):
            self.initialize_dask_cluster()

        self.logger.info(f"THESE ARE GEOG BOUNDS: {self.geog_bounds}")
        # Check to see if we reset geog bounds
        if (
            self.geog_bounds is not None
            and self.geog_bounds["east"] < self.geog_bounds["west"]
        ):
            self.logger.warning(
                f"Requested west longitude of: {self.geog_bounds['west']} and east longitude of {self.geog_bounds['east']}, which spans the prime meridian.  Returning full globe data."
            )
            self.geog_bounds["west"] = None
            self.geog_bounds["east"] = None
            # self.geog_bounds = None

        # First major thing...build the pressure field
        # This will require any of a number of variables.
        # Pressure field may be constructed in many ways
        if self.vcoord_type == "pressure":
            # Data is already in pressure coordinates
            self.get_pressure_from_pressure()
        else:
            # Interpolate to the pressure levels requested
            self.plevels = np.array(plevels)
            if self.vcoord_type == "hybrid-z":
                self.get_pressure_from_hybrid_height()
            elif self.vcoord_type == "hybrid-p":
                self.get_pressure_from_hybrid_press()
        # Make sure our pressure levels are descending
        self.pressure_field = self.pressure_field.sortby(self.vcoord, ascending=False)
        self.logger.info("Done with pressure")
        # Some dimensions
        self.nt, self.nz, self.ny, self.nx = self.pressure_field.shape
        self.startlat = self.pressure_field["lat"][0]
        self.startlon = self.pressure_field["lon"][0]
        # Print some descriptions about the geographic subsetting
        self.logger.info(f"GEOG BOUNDS REQD: {self.geog_bounds}")
        self.logger.info(
            f"ACTUAL LATS: {self.pressure_field['lat'].values[0]} -- {self.pressure_field['lat'].values[-1]}"
        )
        self.logger.info(
            f"ACTUAL LONS: {self.pressure_field['lon'].values[0]} -- {self.pressure_field['lon'].values[-1]}"
        )

        # Some variables needed for the WPS intermediate format
        self.deltalat = abs(
            self.pressure_field["lat"][1] - self.pressure_field["lat"][0]
        )
        self.deltalon = abs(
            self.pressure_field["lon"][1] - self.pressure_field["lon"][0]
        )
        self.earth_radius = 6367470.0 * 0.001
        self.earth_relative_wind = (
            False  # This is only important for lambert projected data
        )
        # Initialize output files to None
        self.WPSfiles = None
        self.source = "Model data"
        self.logger.info("Done with initialization")

    def initialize_dask_cluster(self):
        """
        For some preprocessing options, can be sped up with a Dask cluster enabled
        This initializes a dask cluster
        """
        n_workers = int(cpu_count() / 2)
        self.logger.info(f"Creating Dask cluster with {n_workers} workers")
        self.cluster = LocalCluster(ip="", diagnostics_port=8787, n_workers=n_workers)
        self.client = Client(self.cluster)
        self.terminate_dask_on_complete = True

    def close_dask_cluster(self):
        """
        If we created this cluster, close it
        """
        if self.terminate_dask_on_complete and self.client is not None:
            self.client.close()
            self.cluster.close()
            self.client = None
            self.cluster = None

    def build_wrf_requirements(self):
        """
        Based on the varaible_parameters, assess what needs to be done to build
        a dataset suitable for running WRF
        """
        # These are required variables in CMIP-style
        reqd_types = ["ta", "ua", "va", "hus", "ps", "tsl", "mrlsl", "ts"]
        # These are would-be-nice variables
        would_be_nice = ["tas", "uas", "vas", "huss", "tsk", "psl", "landmask", "snw"]
        # This is what we have
        we_have = list(self.variable_params.index)
        for v in reqd_types:
            if v not in we_have:
                self.logger.error(
                    f"Required variable {v} not found in this data source"
                )
                return
            else:
                self.logger.info(f"Found path to variable {v}")

    def write_to_WPS(self, timedt, field_header, proj_header, field):
        """
        Given an xarray field and the appropriate field_header and proj_header info,
        write this to the appropriate outfile WPS file
        """
        # Grab the outfile from timedt
        outfile = self.WPSfiles[timedt]
        # Write the format header size
        fmtstr = ">i"
        fmtsize = calcsize(fmtstr)
        outfile.write(pack(">i", fmtsize))
        outfile.write(pack(">i", 5))
        outfile.write(pack(">i", fmtsize))

        # Write the main header
        packstr = ">24s f 32s 9s 25s 46s f 3i"
        packsize = calcsize(packstr)
        # packstr ='>char[24],>f,>char[32],>char[9],>char[25],>char[46],>f,>i,>i,>i'
        headerpack = pack(
            packstr,
            bytes(f"{timedt.strftime(field_header['HDATE']): <24}", "utf-8"),
            field_header["XFCST"],
            bytes(f"{field_header['MAP_SOURCE']: <32}", "utf-8"),
            bytes(f"{field_header['FIELD']: <9}", "utf-8"),
            bytes(f"{field_header['UNITS']: <25}", "utf-8"),
            bytes(f"{field_header['DESC']: <46}", "utf-8"),
            field_header["XLVL"],
            field_header["NX"],
            field_header["NY"],
            field_header["IPROJ"],
        )
        # headerpack = [timedt.strftime(field_header['HDATE']), field_header['XFCST'], field_header['MAP_SOURCE'],
        # 	field_header['FIELD'], field_header['UNITS'], field_header['DESC'], field_header['XLVL'],
        # 	field_header['NX'], field_header['NY'], field_header['IPROJ']]
        # outfile.write_record(packstr, *headerpack)
        outfile.write(pack(">i", packsize))
        # print(packsize)
        outfile.write(headerpack)
        outfile.write(pack(">i", packsize))

        # Write the projection header
        # packstr ='>char[8],(5)>f'
        packstr = ">8s 5f"
        packsize = calcsize(packstr)
        projpack = pack(
            packstr,
            bytes(f"{proj_header['STARTLOC']: <8}", "utf-8"),
            proj_header["STARTLAT"],
            proj_header["STARTLON"],
            proj_header["DELTALAT"],
            proj_header["DELTALON"],
            proj_header["EARTH_RADIUS"],
        )
        # projpack = [proj_header['STARTLOC'], proj_header['STARTLAT'],
        # 	proj_header['STARTLON'], proj_header['DELTALAT'], proj_header['DELTALON'],
        # 	proj_header['EARTH_RADIUS']]
        # outfile.write_record(packstr, *projpack)
        outfile.write(pack(">i", packsize))
        outfile.write(projpack)
        outfile.write(pack(">i", packsize))

        # Write the is_wind_relative
        packstr = ">i"
        packsize = calcsize(packstr)
        # print(packsize)
        relpack = pack(packstr, self.earth_relative_wind)
        # outfile.write_record('>?', self.earth_relative_wind)
        outfile.write(pack(">i", packsize))
        outfile.write(relpack)
        outfile.write(pack(">i", packsize))

        # And finally the whole array for this time
        # Select the data for this time and put it in fortran order
        thisarr = np.asfortranarray(field.values)
        arrsize = thisarr.size
        packsize = arrsize * calcsize("f")
        outfile.write(pack(">i", packsize))
        # Write values individually using an iterator over the flattened array
        for v in thisarr.flat:
            outfile.write(pack(">f", v))
        outfile.write(pack(">i", packsize))
        # Flush to file to free up memory
        outfile.flush()
        os.fsync(outfile.fileno())
        # Done
        return

    def add_to_WPS(self, field, soil_levels=None):
        """
        Adds the specified field to the WPS output files associated with this object
        """
        self.logger.info(f"Writing {field.name} to WPS files")
        # First, open up the WPS files if we don't have them already
        if self.WPSfiles is None:
            self.open_WPS_files()

        # Header line is the following:
        # Look up some of this info from the variable table
        field_header = {
            "HDATE": "%Y-%m-%d_%H:%M:%S",
            "XFCST": 0.0,
            "MAP_SOURCE": self.source,
            "FIELD": self.variable_table.loc[field.name]["Metgrid Name"],
            "UNITS": self.variable_table.loc[field.name]["Metgrid Units"],
            "DESC": self.variable_table.loc[field.name]["Description"],
            "XLVL": 100000.0,
            "NX": self.nx,
            "NY": self.ny,
            "IPROJ": self.projection,
        }

        # Projection header
        # Currently assumes cylindrical equidistant projection
        proj_header = {
            "STARTLOC": "SWCORNER",
            "STARTLAT": self.startlat,
            "STARTLON": self.startlon,
            "DELTALAT": self.deltalat,
            "DELTALON": self.deltalon,
            "EARTH_RADIUS": self.earth_radius,
        }

        # Loop through all times
        for tnum, t in enumerate(list(field["time"])):
            # For easy formatting, make this a pure datetime object
            # print("On Time:", t.values)
            #print(" RGF current field ",field_header["FIELD"])
            timedt = pd.to_datetime(str(t.values))
            self.logger.info(f"   {timedt}")
            # Three possibilities here ... single level, pressure levels, or soil levels
            # RGF mod.. do 2mT, RH, 10m wind first. NEEDED TO GET LEVEL 200100 FOR THESE VARIABLES given they have same metgrid names as isobaric level fields
            if field.name in ["VAR_2T", "VAR_10U", "VAR_10V", "VAR_2D","VAR_2RH"]: # "VAR_2D" should not be used anymore
                print(" RGF *** we have 2m T or Td or RH or 10 m wind")
                field_header["XLVL"] = 200100.0
                # Write to file
                self.write_to_WPS(
                    timedt, field_header, proj_header, field[{self.timecoord: tnum}]
                )
                
            elif self.vcoord in field.coords:
                #print(" *** RGF isobaric levels variable ",field.coords,self.vcoord)
                # Loop through pressure levels
                for lnum, lev in enumerate(list(field[self.vcoord].values)):
                    #print("      ",lnum, lev)
                    # Make the header reflect this level
                    field_header["XLVL"] = float(lev)*100 # RGF FORCE to PASCALS -- this was a KEY FIX **********************
                    the_mean=np.mean(field[{self.timecoord: tnum, self.vcoord: lnum}])
                    #print(" RGF field mean for lev ",lev,the_mean.values)
                    # Write to file
                    self.write_to_WPS(
                        timedt,
                        field_header,
                        proj_header,
                        field[{self.timecoord: tnum, self.vcoord: lnum}],
                    )
            elif (field.name in ["tsoil", "soilw", "mrlsl", "soilt", "tsl"]) or (
                "depth" in field.coords
            ):
                # Loop through soil depths
                # print("Adding soil variable to WPS")
                print(" *** RGF depth variable") # this code is never called with ERA5 soil variable names
                for lnum, lev in enumerate(list(field[self.soilcoord].values)):
                    # Make the header reflect this level -> The same for soil depths
                    # print("      ",lnum, lev)
                    field_header["XLVL"] = 200100.0
                    top_lev = int(field["top"].values[lnum])
                    bot_lev = int(field["bot"].values[lnum])
                    # Field name needs to reflect the bounds of this layer
                    if field.name in ["tsl", "tsoil", "TSOI"]:
                        field_header["FIELD"] = "ST{:03d}{:03d}".format(
                            top_lev, bot_lev
                        )
                        field_header[
                            "DESC"
                        ] = "Soil Temperature {:d}-{:d}cm Layer".format(
                            top_lev, bot_lev
                        )
                    elif field.name in ["mrlsl", "soilw", "SOILLIQ"]:
                        field_header["FIELD"] = "SM{:03d}{:03d}".format(
                            top_lev, bot_lev
                        )
                        field_header["DESC"] = "Soil Moisture {:d}-{:d}cm Layer".format(
                            top_lev, bot_lev
                        )
                    # print(field_header)
                    # Write to file
                    self.write_to_WPS(
                        timedt,
                        field_header,
                        proj_header,
                        field[{self.timecoord: tnum, self.soilcoord: lnum}],
                    )

            else:
                # This is a single field.  Includes all soil variables
                print(" *** RGF single level variable",field_header["FIELD"])
                if field_header["FIELD"] == "psl" or field_header["FIELD"] == "PMSL": # RGF MOD added 'PMSL'
                    field_header["XLVL"] = 201300.0
                    #print(" RGF modified level to 201300.00")
                else:
                    # Must be at surface
                    field_header["XLVL"] = 200100.0
                # Write to file
                self.write_to_WPS(
                    timedt, field_header, proj_header, field[{self.timecoord: tnum}]
                )

    def open_WPS_files(self, outpath="."):
        # Need a file for each valid time
        # if self.timecoord != 'hour':
        try:
            valid_times = list(self.pressure_field["time"].values)
        except KeyError:
            valid_times = list(
                pd.date_range(
                    self.time_range[0], self.time_range[-1], freq=self.time_freq
                )
            )
        timefmt = [pd.to_datetime(str(x)) for x in valid_times]
        # print(type(timefmt[0]))
        # self.WPSfiles = {t : FortranFile(f'FILE:{tfmt:%Y-%m-%d_%H}', 'w') for t, tfmt in zip(valid_times, timefmt)}
        self.WPSfiles = {
            tfmt: open(f"{outpath}/FILE:{tfmt:%Y-%m-%d_%H}", "wb")
            for t, tfmt in zip(valid_times, timefmt)
        }
        return

    def close_WPS_files(self):
        """
        Closes all open WPS files
        """
        if self.WPSfiles is None:
            return
        for t, f in self.WPSfiles.items():
            # Close the file
            f.close()
        # Set this to none
        self.WPSfiles = None
        return

    def load_and_subset(self, var_name="ta", filepath=None):
        """
        Goes through the sequence to find files, load them and subset in time
        """
        self.logger.info(f"Loading and subsetting {var_name}")
        # try:
        if filepath is None:
            if var_name in self.variable_paths:
                # Use the hard-coded variable path here
                filepath = self.variable_paths[var_name]
                # Change the variable name if needed
                try:
                    var_name = self.variable_names[var_name]
                except:
                    pass

            elif self.master_path.startswith("http"):
                # This is a THREDDS served file
                # Assume we have a placeholder in master_path
                # Figure out if we span certain limits in time (files done by decade or year)
                if "{decade}" in self.master_path:
                    # Do we span decades?
                    start_decade = int(np.floor(self.time_range[0].year / 10) * 10)
                    end_decade = int(np.floor(self.time_range[1].year / 10) * 10)
                    if start_decade != end_decade:
                        # going to make an assumption here we aren't doing runs longer than 10 years...
                        filepath = [
                            self.master_path.format(
                                **{"variable": var_name, "decade": start_decade}
                            ),
                            self.master_path.format(
                                **{"variable": var_name, "decade": end_decade}
                            ),
                        ]
                    else:
                        filepath = self.master_path.format(
                            **{"variable": var_name, "decade": start_decade}
                        )
                elif "{year}" in self.master_path:
                    # Do we span years?
                    start_year = self.time_range[0].year
                    end_year = self.time_range[1].year
                    if start_year != end_year:
                        # going to make an assumption here we aren't doing runs longer than 2 years...
                        filepath = [
                            self.master_path.format(
                                **{"variable": var_name, "year": start_year}
                            ),
                            self.master_path.format(
                                **{"variable": var_name, "year": end_year}
                            ),
                        ]
                    else:
                        filepath = self.master_path.format(
                            **{"variable": var_name, "year": start_year}
                        )
                elif "{var_type}" in self.master_path:
                    # This is likely just a ta file that needs to be upper air
                    filepath = self.master_path.format(**{"var_type": "upa"})
                else:
                    # Just one file, hopefully.  Possibly with date info if it's something weird
                    filepath = self.master_path.format(
                        **{"variable": var_name, "start_date": self.time_range[0]}
                    )
                # Extract the subset
            else:
                # This is a local file
                # Find the files that match this variable name
                filepath = self.find_overlapping_file(var_name)
            self.logger.info(f"Loading variable {var_name} from: {filepath}")
            data_subset = self.extract_subset(filepath)
        self.logger.info
        # Check on the variable name here
        try:
            outarray = self.resample_in_time(data_subset[var_name])
        except KeyError:
            outarray = self.resample_in_time(
                data_subset[alternate_variable_names[var_name]]
            )
            # Be sure we come out with the correct name
            outarray.name = var_name
        
        # RGF convert ERA5 geopotential field to geopotential height
        if(var_name=="Z"):
            outarray = outarray/9.80665
            print(" RGF converted geopotential to geopotential height (m)")
        #
        return outarray

    def find_overlapping_file(self, var_name="ta"):
        """
        Will search the master_path for variable files matching this 
        variable (and, in the future, overlapping in time) and return them
        """
        # Simple search now...just find files that start with the variable name
        format_path = self.master_path.format(
            **{"variable": var_name, "start_date": self.time_range[0]}
        )
        if format_path.endswith(".nc") or self.infile_format == "zarr":
            # This is a single file
            filepath = format_path
        else:
            allfiles = [
                f for f in os.listdir(format_path) if f.startswith(var_name + "_")
            ]
            filepath = os.path.join(self.master_path, allfiles[0])
        return filepath

    def get_pressure_from_pressure(self):
        """
        For datasets already in pressure coordinates, just need to populate the relevant info
        about this dataset
        """
        self.logger.info("Building domain structure from pressure field")
        # Load in air temperature
        ta_subset = self.load_and_subset(var_name="T") # RGF was hardcoded as "ta". Need to change to "T"
        # print("Done with load and subset")
        # Get the pressure coordinates
        # print(ta_subset)
        plevs = ta_subset[self.vcoord].values
        # Make sure this is in the correct order
        if plevs[1] > plevs[0]:
            plevs = plevs[::-1]
        # Make a data array for this
        p_field = ta_subset.copy(deep=True)
        p_field.name = "pressure"
        p_field.attrs["standard_name"] = "air_pressure"
        p_field.attrs["long_name"] = "Air Pressure"
        p_field.attrs["units"] = "Pa"
        p_field.values[:] = np.ones(p_field.shape)
        # Basically copy over all the pressure levels
        # Check to be sure we are actually in Pa
        #print(" RGF plevs ",plevs)
        if plevs.max() < 85000:
            self.logger.warning("Suspect we have hPa...converting to Pa")
            # Multiply by 100
            plevs = plevs * 100
            #print(" RGF plevs after ",plevs)
        p_field.values[:] = plevs[None, :, None, None] * p_field[:]
        # print('setting plevels')
        self.plevels = list(plevs)
        #print(" RGF ** PLEVELS ",self.plevels)
        #print(" RGF self.vcoord ",self.vcoord)
        # print('setting p_field')
        self.pressure_field = p_field
        # print('Done with pressure build')

    def get_pressure_from_hybrid_height(self):
        """
        Build the pressure field in files with hybrid-z coordinates
        WILL REQURE the ps field and ta field, so it will search for those
        """
        self.logger.info("Building full pressure field from hybrid coordinates")
        self.logger.info(
            "Subsetting surface pressure and temperature for full pressure field"
        )
        # We need ps first
        fname = "ps"
        filepath = self.find_overlapping_file(fname)
        ps_subset = self.extract_subset(filepath)
        # Resample in time
        # ps_subset = self.resample_in_time(ps_subset)
        # Then ta for the full 3d grid info
        fname = "ta"
        filepath = self.find_overlapping_file(fname)
        ta_subset = self.extract_subset(filepath)
        # Resample in time here
        # ta_subset = self.resample_in_time(ta_subset)
        # Construct the full pressure field from the info in these files
        self.convert_hybrid_z_to_pressure(ps_subset, ta_subset)
        # Resample in time
        self.pressure_field = self.resample_in_time(self.pressure_field)

    def get_pressure_from_hybrid_press(self):
        """
        Build the pressure field in files with hybrid-p coordinates
        WILL REQUIRE the ta field set as master_path
        CESM-LENS will need the ps_path set as well
        """
        self.logger.info("Building full pressure field from file")
        # Use the ta field...it seems common across most datasets
        filepath = self.master_path
        ta_subset = self.extract_subset(filepath)
        # Resample in time
        # ta_subset = self.resample_in_time(ta_subset)
        # For CESM-LENS-like variable, need to get ps as well
        if "ps" not in list(ta_subset.variables.keys()):
            ps_subset = self.extract_subset(self.ps_path)
            # Insert the surface pressure field in the ta subset
            try:
                # Construct the full pressure field
                self.convert_hybrid_p_to_pressure(ta_subset, ps_subset["ps"])
            except KeyError:
                self.convert_hybrid_p_to_pressure(ta_subset, ps_subset["PS"])
        else:
            self.convert_hybrid_p_to_pressure(ta_subset)
        # Resample in time
        self.pressure_field = self.resample_in_time(self.pressure_field)

    def subset_time_indices(self, filename):
        """
        Currently, finds the indexes along each dimension to subset
        Searches for all times in the file that span the start/end dates requested
        """
        start, end = self.time_range
        start = np.datetime64(start)
        end = np.datetime64(end)
        #print(" RGF start, end after ",start,end)
        # with Dataset(filename, 'r') as dset:
        # if filename.startswith('http'):
        # 	engine = 'pydap'
        # else:
        # 	engine = 'netcdf4'
        engine = "netcdf4"
        if True:
            # Try first with the netcdf4 engine, otherwise switch to pydap (Slower, but more robust)
            try:
                if isinstance(filename, str):
                    if self.infile_format == "zarr":
                        mapping = s3fs.S3Map(filename, s3=s3fs.S3FileSystem())
                        # Try to open with consolidated metadata first for speed
                        try:
                            dset = xarray.open_zarr(mapping, consolidated=True)
                            self.logger.debug("Using consolidated Zarr metadata")
                        except:
                            dset = xarray.open_zarr(mapping)
                            self.logger.debug("Using standard Zarr metadata")
                    elif self.infile_format == "netCDF":
                        dset = xarray.open_dataset(filename)
                else:
                    dset = xarray.open_mfdataset(filename)
            except OSError:
                self.logger.warning(
                    "Slow load (likely large file)...switching to pydap engine"
                )
                engine = "pydap"
                if isinstance(filename, str):
                    if self.infile_format == "zarr":
                        mapping = s3fs.S3Map(filename, s3=s3fs.S3FileSystem())
                        # Try to open with consolidated metadata first for speed
                        try:
                            dset = xarray.open_zarr(mapping, consolidated=True)
                            self.logger.debug("Using consolidated Zarr metadata")
                        except:
                            dset = xarray.open_zarr(mapping)
                            self.logger.debug("Using standard Zarr metadata")
                    elif self.infile_format == "netCDF":
                        dset = xarray.open_dataset(filename, engine=engine)
                else:
                    dset = xarray.open_mfdataset(filename, engine=engine)

            # If there is only one time in this file, assume it is a static field
            if len(dset.variables[self.timecoord]) == 1:
                self.logger.info("Assuming this is a static field")
                return 0, 1
            # Find the time overlap, sorting out crazy netcdftime calendars
            # starttime = cdftime.num2date(cdftime.date2num(start))
            # endtime = cdftime.num2date(cdftime.date2num(end))

            times = dset[self.timecoord] 
            #print(" RGF times ",times)  
            use_times = times[np.bitwise_and(times >= start, times <= end)]
            #print(" RGF use_times ",use_times)
            if use_times.shape[0] == 0:
                # If no times found, then get the data that bounds the time
                end_index = int(np.argmax(times >= end) + 1)
                start_index = int(end_index - 3)
            else:
                # all_indices = times.values.argsort()
                # use_indices = [list(times.values).index(t) for t in use_times]
                use_indices = np.nonzero(np.in1d(times, use_times))[0]
                start_index = int(use_indices[0])
                end_index = int(use_indices[-1])
                #print(" RGF initial indices ",start_index,end_index)
                #print(" RGF use_times[0], start ",use_times[0],start)
                # Double check to be sure we full span the range
                if use_times[0] > start:
                    start_index -= 1
                if use_times[-1] < end:
                    end_index += 1
        dset.close()
        # Because of Python indexing, add 1 to end_index
        end_index += 1
        #print(" RGF return from subset_time_indices ",start_index,end_index)
        return start_index, end_index

    def extract_subset(self, filename):
        """
        Extract a subset in time and space
        """
        # First, get the time indices that surround this with a raw netCDF call
        # to correctly handle the calendar
        if self.timecoord != "hour":
            time_indices = self.subset_time_indices(filename)
        # Set the slices for
        # Now load as an xarray
        # Figure out the engine; pydap is more robust for dap
        # if filename.startswith('http'):
        # 	engine = 'pydap'
        # else:
        # 	engine = 'netcdf4'
        engine = "netcdf4"
        multifile = False
        try:
            if isinstance(filename, str):
                if self.infile_format == "zarr":
                    mapping = s3fs.S3Map(filename, s3=s3fs.S3FileSystem())
                    # Try to open with consolidated metadata first for speed
                    try:
                        fulldset = xarray.open_zarr(mapping, consolidated=True)
                        self.logger.debug("Using consolidated Zarr metadata")
                    except:
                        fulldset = xarray.open_zarr(mapping)
                        self.logger.debug(
                            "Using standard (non-consolidated) Zarr metadata"
                        )
                elif self.infile_format == "netCDF":
                    # Just one file
                    #fulldset = xarray.open_dataset(filename, engine=engine)
                    fulldset = xarray.load_dataset(filename, engine=engine) # RGF load_dataset instead
            else:
                # Multiple files
                fulldset = xarray.open_mfdataset(filename, engine=engine)
                multifile = True
        except OSError:
            engine = "pydap"
            if isinstance(filename, str):
                # Just one file
                fulldset = xarray.open_dataset(filename, engine=engine)
            else:
                # Multiple files
                fulldset = xarray.open_mfdataset(filename, engine=engine)
                multifile = True
        except ValueError:
            if isinstance(filename, str):
                # Just one file
                fulldset = xarray.open_dataset(filename, engine=engine, decode_cf=False)
            else:
                # Multiple files
                fulldset = xarray.open_mfdataset(
                    filename, engine=engine, decode_cf=False
                )
                multifile = True
            fulldset = xarray.open_dataset(filename, decode_cf=False, engine=engine)
            for var in list(fulldset.variables.keys()):
                if "_FillValue" in list(fulldset[var].attrs.keys()):
                    del fulldset[var].attrs["_FillValue"]
            fulldset = xarray.conventions.decode_cf(fulldset)
        # print(fulldset)

        # Do a quick renaming here if the lats and lons are misnamed
        if "lon" not in list(fulldset.variables.keys()):
            fulldset = fulldset.rename({"longitude": "lon", "latitude": "lat"})

        # Get the bounding times
        slicer = {}
        # Use the indices from subset_time_indices
        if self.timecoord != "hour":
            slicer[self.timecoord] = slice(time_indices[0], time_indices[1])
        if self.geog_bounds is not None:
            # Check to be sure the dataset supports this range
            if (
                self.geog_bounds["east"] is not None
                and self.geog_bounds["west"] is not None
            ):
                if (fulldset["lon"].max() < self.geog_bounds["east"]) or (
                    fulldset["lon"].min() > self.geog_bounds["west"]
                ):
                    self.logger.error(
                        "Requested domain longitude bounds outside of range possible from this data source"
                    )
                    self.logger.info(
                        f"This data source longitude range: {float(fulldset['lon'].min())} --  {float(fulldset['lon'].max())}"
                    )
                    self.logger.info(
                        f"Requested longitude range:        {self.geog_bounds['west']} -- {self.geog_bounds['east']}"
                    )
                    return None
                west_index = int(
                    np.argmin(np.abs(fulldset["lon"] - self.geog_bounds["west"])).values
                )
                east_index = int(
                    np.argmin(np.abs(fulldset["lon"] - self.geog_bounds["east"])).values
                )
                slicer["lon"] = slice(
                    min(west_index, east_index) - 2, max(west_index, east_index) + 2
                )
            else:
                # We were given "None" for one of the longitude bounds, so return the whole globe
                pass
            if (fulldset["lat"].max() < self.geog_bounds["north"]) or (
                fulldset["lat"].min() > self.geog_bounds["south"]
            ):
                self.logger.error(
                    "Requested domain latitude bounds outside of range possible from this data source"
                )
                self.logger.info(
                    f"This data source latitude range: {float(fulldset['lat'].min())} -- {float(fulldset['lat'].max())}"
                )
                self.logger.info(
                    f"Requested latitude range:        {self.geog_bounds['south']} -- {self.geog_bounds['north']}"
                )
                return None
            # Get space indices
            south_index = int(
                np.argmin(np.abs(fulldset["lat"] - self.geog_bounds["south"])).values
            )
            north_index = int(
                np.argmin(np.abs(fulldset["lat"] - self.geog_bounds["north"])).values
            )
            # This weird syntax is in case the lats/lons are not in ascending order
            # (it happens...)
            # Add 2 gridpoints for padding on all sides
            slicer["lat"] = slice(
                min(south_index, north_index) - 2, max(south_index, north_index) + 2
            )

        # Subset the data in time
        subd = fulldset[slicer]
        # Make sure now that the latitudes are ascending (can't do on full dataset...memory errors?)
        subd = subd.sortby("lat", ascending=True)
        # Decode the times here
        subd = xarray.decode_cf(subd)
        # If this is a multi-file dataset, force load
        if multifile:
            self.logger.info("Loading multifile dataset")
            subd = subd.load()
        return subd

    def convert_hybrid_z_to_pressure(self, ps_field, ta_field):
        """
        Computes the full pressure field given the surface pressure (ps)
        and 3d temperature (ta)
        CHECK TO BE SURE LOWEST LEVEL IS NOT PSFC
        """
        # Convert to pressure levels
        # Formula for model hybrid levs
        z_full = ta_field[self.vcoord] + ta_field["b"] * ta_field["orog"]

        # Now we need to convert to pressure levels
        p_field = ta_field["ta"].copy(deep=True)
        p_field.name = "pressure"
        p_field.attrs["standard_name"] = "air_pressure"
        p_field.attrs["long_name"] = "Air Pressure"
        p_field.attrs["units"] = "Pa"
        # Lowest level must be built from surface using hydrostatic/hypsometric
        p_field[:, 0, :, :] = ps_field["ps"][:, :, :] * np.exp(
            -(9.81 / 287.0 / (ta_field["ta"].isel(lev=0)))
            * (z_full.isel(lev=0) - ta_field["orog"])
        )
        # p_field[:,0,:,:] = ps_field['ps'][:,:,:]
        # Reconstruct full pressure field using hydrostatic/hypsometric
        for ldex in range(1, z_full.shape[0]):
            p_field[:, ldex] = p_field.isel(lev=ldex - 1) * np.exp(
                -(
                    9.81
                    / 287.0
                    / (
                        (
                            ta_field["ta"].isel(lev=ldex)
                            + ta_field["ta"].isel(lev=ldex - 1)
                        )
                        / 2.0
                    )
                )
                * (z_full.isel(lev=ldex) - z_full.isel(lev=ldex - 1))
            )
        # Make this a data array
        # p_field = xarray.DataArray(p_field, coords=ta_field.coords, dims=ta_field.dims)
        self.pressure_field = p_field

    def convert_hybrid_p_to_pressure(self, ta_field, ps_field=None):
        """
        This is for models that specify their hybrid levels in full pressure (not height) coords
        """
        # See which pattern of coefficient variables we have
        if "a" in list(ta_field.variables.keys()):
            if ps_field is not None:
                p_full = ta_field["a"] * ta_field["p0"] + ta_field["b"] * ps_field
            else:
                p_full = ta_field["a"] * ta_field["p0"] + ta_field["b"] * ta_field["ps"]
        elif "hyam" in list(ta_field.variables.keys()):
            # CESM-LENS formatted
            if ps_field is not None:
                p_full = ta_field["hyam"] * ta_field["P0"] + ta_field["hybm"] * ps_field
            else:
                p_full = (
                    ta_field["hyam"] * ta_field["P0"]
                    + ta_field["hybm"] * ta_field["ps"]
                )

        # Make sure here that time is the leading dimension
        # print(list(p_full.coords))
        if list(p_full.coords).index(self.timecoord) != 0:
            p_full = p_full.transpose(self.timecoord, self.vcoord, "lat", "lon")
        self.pressure_field = p_full

    def process_soil_levels(self, varray, ismoist=False, interpolate_to=None): # not used in current version
        """
        Figures out the soil levels in the file
        And does unit conversions if needed
        """
        # Assumes we are given a depth dataarray
        # Check the units on the depths
        # If in meters, convert to cm
        # For now, hardcode the CESM (levgrnd) as needing conversion
        if (varray[self.soilcoord].max() < 10) or (self.soilcoord == "levgrnd"):
            self.logger.info("Converting soil depth from meters to cm")
            # This is in meters
            attrs = varray.attrs
            varray["new depth"] = varray[self.soilcoord] * 100.0
            varray = varray.drop(self.soilcoord)
            varray[self.soilcoord] = varray["new depth"][:]
            varray = varray.drop("new depth")
            varray.coords[self.soilcoord] = varray[self.soilcoord]
            varray.attrs = attrs
        # WRF wants soil moisture in m3/m3 for soil
        if (varray.name in ["mrlsl", "soilm", "SOILLIQ", "soilw"]) and (
            varray.units not in ["frac", "fraction", "frac."]
        ):
            # Assume value reported is constant for layer above
            # Based on examples at:
            # https://wolfscie.wordpress.com/2017/05/05/working-with-cmip5-data-in-wrf-1/
            # http://www.meteo.unican.es/wiki/cordexwrf/SoftwareTools/CmorPreprocessor
            if varray.units in ["kg m-2", "kg/m2"]:
                self.logger.info(
                    f"Converting soil moisture from {varray.units} to fraction"
                )
                # Compute the ratio
                depth_bounds = list(varray[self.soilcoord].values)
                for bnum, bound in enumerate(depth_bounds):
                    if bnum == 0:
                        layer_depth = bound
                    else:
                        layer_depth = bound - depth_bounds[bnum - 1]
                    # Figure out the weight based on the thickness of this layer
                    # 1000 kg H2O / m^3; 100 cm / m
                    weight = 1.0 / (1000.0 * (layer_depth / 100.0))
                    # Multiply by this weight
                    varray[:, bnum] = varray[:, bnum] * weight
                varray.attrs["units"] = "fraction"

        # Interpolate to levels
        if interpolate_to is None:
            nt, nz, ny, nx = varray.shape
            soil_levels = varray[self.soilcoord].values
            nlevs = soil_levels.shape[0]
            # nlevs = len(self.soil_levels) - 1
            lev_interp = np.ndarray(shape=(nt, nlevs, ny, nx))
            tops = []
            bottoms = []
            for lev in range(0, nlevs):
                # Need integers for levels...hence the round
                if lev == 0:
                    tops.append(0)
                    bottoms.append(round(soil_levels[lev]))
                    # Average for now
                    lev_interp[:, 0, :, :] = varray[:, lev]
                else:
                    tops.append(round(soil_levels[lev - 1]))
                    bottoms.append(round(soil_levels[lev]))
                    # Average for now
                    lev_interp[:, lev, :, :] = 0.5 * (
                        varray[:, lev] + varray[:, lev - 1]
                    )
        else:
            # Use the provided interpolation levels
            nt, nz, ny, nx = varray.shape
            # Get this from interpolate_to argument
            if interpolate_to[0] == 0:
                interpolate_to = interpolate_to[1:]
            soil_levels = np.array(interpolate_to)
            actual_levels = varray[self.soilcoord].values
            nlevs = soil_levels.shape[0]
            self.logger.info(f"Interpolating to levels: {soil_levels}")
            lev_interp = np.ndarray(shape=(nt, nlevs, ny, nx))
            tops = []
            bottoms = []
            for lev in range(0, nlevs):
                if lev == 0:
                    topval = 0
                    botval = round(soil_levels[lev])
                    # Make first level mean value actually equal the bottom of the level
                    meanval = botval
                else:
                    topval = round(soil_levels[lev - 1])
                    botval = round(soil_levels[lev])
                    # Find the mean level
                    meanval = 0.5 * (topval + botval)
                # distance-weighted mean
                if np.isin(meanval, actual_levels):
                    # Weights are 1 there and 0 everywhere else
                    weights = np.where(actual_levels == meanval, 1, 0)
                else:
                    # Inverse distance weight
                    diff = 1.0 / (np.abs(actual_levels - meanval) ** 1)
                    weights = diff / diff.sum()
                # Compute mean and insert
                lev_interp[:, lev] = (varray * weights[None, :, None, None]).sum(axis=1)
                # Add the levels too
                tops.append(topval)
                bottoms.append(botval)

        # Make this into an output variable array
        # New axis of depth
        coords = {}
        coords[self.timecoord] = varray.coords[self.timecoord]
        coords["lat"] = varray.coords["lat"]
        coords["lon"] = varray.coords["lon"]
        coords["top"] = (self.soilcoord, tops)
        coords["bot"] = (self.soilcoord, bottoms)
        coords["depth"] = (self.soilcoord, 0.5 * (np.array(tops) + np.array(bottoms)))
        # LEM -> revert self.soilcoord to 'depth' if this fails
        outarray = xarray.DataArray(
            lev_interp,
            dims=(self.timecoord, self.soilcoord, "lat", "lon"),
            coords=coords,
            name=varray.name,
            attrs=varray.attrs,
        )
        return outarray

    def interp_to_p_levels(self, varray, plevs=None):
        """
        Interpolate to requested pressure levels in Pa
        """
        self.logger.info("Interpolating to pressure levels")
        if plevs is None:
            plevs = self.plevels
        # print("Using plevs:", plevs)
        # Make sure our vertical levels are DESCENDING with height
        varray = varray.sortby(self.vcoord, ascending=False)
        # Attach the pressure field as a coordinate to varray
        varray.coords["pressure"] = (
            ("time", self.vcoord, "lat", "lon"),
            self.pressure_field,
        )
        # Build a new array with different vertical dimension
        use_shape = list(varray.shape)
        # Make the vertical dimension (lev) in the new array have
        # the specified number of pressure levels
        use_shape[1] = plevs.shape[0]
        # Set all the other dimensions/coordinates directly from the
        # original array
        use_dims = {
            "time": varray["time"].shape[0],
            self.vcoord: plevs.shape[0],
            "lat": varray["lat"].shape[0],
            "lon": varray["lon"].shape[0],
        }
        use_coords = {
            "time": varray["time"].values,
            self.vcoord: plevs,
            "lat": varray["lat"].values,
            "lon": varray["lon"].values,
        }
        # Make an empty array to populate with the interpolated data
        interpolated = np.zeros(use_shape)
        interpolated[:] = np.nan
        # print(varray)
        # print(interpolated)
        # Loop through all requested pressure levels
        # TODO: Can this be vectorized somehow?
        for pdex, plevel in enumerate(list(plevs)):
            # Get the nearest pressure level data within 200 mb (20000 Pa)
            this = varray.where(np.abs((varray.pressure - plevel)) < 20000)
            pmask = varray["pressure"].where(np.abs((varray.pressure - plevel)) < 20000)
            # Inverse distance weighting...here it's linear in distance
            # TODO: Can raise to a power other than one to emphasize closer
            # pressure levels
            pdiff_inv = 1.0 / (np.abs(pmask - plevel) ** 3)
            pweights = pdiff_inv / pdiff_inv.sum(dim=self.vcoord)
            # Multiply the selected levels by the weights and sum
            field = (pweights * this).sum(dim=self.vcoord).values
            # print(field)
            # Going back to the original pressure field, find where
            # the maximum pressure (that is, the pressure at ground level)
            # was still less (above) the pressure level we are requesting.
            # This means that the pressure level at those locations is
            # actually underground.  Make these NAN for now
            thefilter = (varray.pressure.max(dim=self.vcoord) < plevel).values
            field[thefilter] = np.nan
            # Fill in our empty array
            interpolated[:, pdex, :, :] = field
            # print(this)
            # interp_field = varray.sel(pressure=plevel, method='nearest', tolerance=5000)
            # interpolated[:,pdex,:,:] = interp_field.values

        # Convert the new empty array to an xarray with the appropriate dimensions
        interpolated = xarray.DataArray(
            interpolated, dims=use_dims, coords=use_coords, name=varray.name
        )

        return interpolated

    def fill_underground_ecmwf(self, varray):
        """
        Use ECMWF method to fill in 3D fields where below terrain
        https://opensky.ucar.edu/islandora/object/technotes%3A168
        """
        self.logger.info("Filling in below ground following the ECWMF method")
        # Goal is to find the lowest value that is not nan
        # Rename the vertical coordinate if needed
        if self.vcoord not in list(varray.dims):
            varray = varray.rename({"level": self.vcoord})
        # Start by making the vertical coordinate last for convenience
        varray = varray.transpose(self.timecoord, "lat", "lon", self.vcoord)
        # Check here to be sure that the pressure coordinate
        # is DESCENDING
        varray = varray.sortby(self.vcoord, ascending=False)

        # This finds the index of the first non-nan value along
        # the vertical coordinate
        pdex = np.argmax(np.isfinite(varray), axis=-1).values
        # Find the highest level that still has underground points for later
        max_underground_lev = pdex.max(axis=None)
        # Flatten out our data array to simplify the indexing
        # print("VARRAY SHAPE:", varray.shape)
        # print("LEN PLEVS:", len(self.plevels))
        # Check to see if we cut off the top levels
        lowest_values = varray.values[:, :, :, : len(self.plevels)].reshape(
            -1, len(self.plevels)
        )
        # Flatten the index array as well
        flatpdex = pdex.flatten()
        # Grab the lowest values along the vertical dimension
        lowest_values = lowest_values[range(flatpdex.shape[0]), flatpdex]
        lowest_plevs = np.take(varray[self.vcoord].values, flatpdex)
        # Reshape these back
        lowest_values = lowest_values.reshape(self.nt, self.ny, self.nx)
        lowest_plevs = lowest_plevs.reshape(self.nt, self.ny, self.nx)
        # And reset the dimensions of the variable
        varray = varray.transpose(self.timecoord, self.vcoord, "lat", "lon")
        # Now we have to figure out how to do the interpolation.  Only
        # special for pressure, otherwise we just repeat the lowest
        # value below ground
        if varray.name in ["t", "ta"]:
            # Do the temperature method
            # Standard lapse rate
            alpha = 0.0065 * (287.0 / 9.81)
            # Need psfc
            for lev in range(max_underground_lev):
                # Ratio of current pressure to lowest pressure level
                prat = alpha * np.log(float(varray[self.vcoord][lev]) / lowest_plevs)
                Tproxy = lowest_values * (
                    1.0 + prat + 0.5 * prat ** 2 + 1.0 / 6.0 * prat ** 3
                )
                # Fill where underground
                thislev = varray[:, lev, :, :]
                under = np.isnan(thislev).values
                # Set these places to be the Tproxy at this place
                thislev = np.where(under, Tproxy, thislev)
                varray[:, lev, :, :] = thislev
        else:
            # Do the repitition method
            # Loop through the levels and repopulate with duplicated data
            for lev in range(max_underground_lev):
                # Where are we underground
                thislev = varray[:, lev, :, :]
                under = np.isnan(thislev).values
                # Set these places to be the lowest_value at this place
                thislev = np.where(under, lowest_values, thislev)
                varray[:, lev, :, :] = thislev
        return varray

    def resample_in_time(self, varray, power=2):
        """
        Given an xarray datarray with times that span those in time_range, interpolate
        to that time range with the given frequency
        power -> How rapidly we should decay from a time
        """
        start, end = self.time_range
        # If this is a static field (length of time dim is 1): just make copies
        # at start and end time, overwriting original array
        if varray[self.timecoord].shape[0] == 1:
            self.logger.info("Duplicating static field for all times")
            new_times = self.time_range
            dshape = list(varray.shape)
            # 2 times...start and end
            dshape[0] = 2
            # make an empty array
            new_arr = np.zeros(dshape)
            # Populate it with the static field twice
            new_arr[0] = varray[0].values
            new_arr[1] = varray[0].values
            # Rebuild the data array
            new_coords = dict(varray.coords)
            new_coords[self.timecoord] = [start, end]
            # Overwrite exiting varray
            varray = xarray.DataArray(
                new_arr,
                coords=new_coords,
                dims=varray.dims,
                name=varray.name,
                attrs=varray.attrs,
            )
        elif self.timecoord == "hour":
            # Don't need to resample
            return varray
        else:
            # Not a static file
            # Test to see whether this actually needs to be done
            dt = int(varray[self.timecoord][1] - varray[self.timecoord][0])
            start_time = self.dt64_to_datetime(varray[self.timecoord][0].values)

            # See if this matches the requested time_freq
            time_freq_hrs = int(self.time_freq[:-1])
            if (time_freq_hrs == int(dt / 3.6e12)) and (
                start_time == self.time_range[0]
            ):
                self.logger.info(
                    f"Data already at requested frequency of {self.time_freq}"
                )
                return varray

            self.logger.info("Resampling in time to requested frequency")

        # Reduce to just the period of interest after resampling in time
        # Default resampling just will populate with the values at the nearest time
        # resampled = varray.resample(freq=self.time_freq, dim=self.timecoord)
        # Load if dask
        varray = varray.load()
        resampled = varray.resample(**{self.timecoord: self.time_freq}).interpolate(
            "linear"
        )
        # Meet back up here for the rest
        resampled = resampled.loc[
            {
                self.timecoord: slice(
                    start.strftime("%Y-%m-%dT%H"), end.strftime("%Y-%m-%dT%H")
                )
            }
        ]
        # We want to actually interpolate to times instead of
        # just using the values at the nearest time in the record
        # Do this through inverse distance weighting
        # Find the distance between each time and each target time
        time_diffs = [
            abs(varray[self.timecoord] - x).values.astype(np.float64)
            for x in list(resampled[self.timecoord])
        ]
        # Inv distance weight
        # At each point, either return 1/x as weight or set the matching value to 1 if exact match
        # "power" can be adjusted to more heavily weight values closer to the target time
        weight_inv = np.array(
            [
                1.0 / (x ** power) if x.min() != 0.0 else np.where(x == 0.0, 1.0, 0.0)
                for x in time_diffs
            ]
        )
        # Divide by the sum
        weight_inv = weight_inv / weight_inv.sum(axis=1)[:, None]
        # Now do the mapping with the weights
        # print(weight_inv)
        # Now, in the stupidest thing I've found in Python, we have to use
        # tensordot instead of just dot because of dimension ordering
        # print(weight_inv.shape, varray.shape)
        resampled[:] = np.tensordot(weight_inv, varray, axes=1)
        resampled.attrs = varray.attrs
        return resampled

    def dt64_to_datetime(self, dt64):
        """
        Convert datetime64 object to datetime
        """
        #print(" RGF dt64 ",dt64, dt64.dtype)
        ts = (dt64 - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
        return datetime.utcfromtimestamp(ts)

    def regrid_external_dset(self, varray, vcoord=None, method="bilinear"):
        """
        Given a variable array on a different grid, invoke the ESMF regridder to put this
        on the same grid 
        """
        # Build the regridder
        # Figure out where we are finite in the vertical
        if len(varray.shape) == 4:
            use_areas = np.isfinite(np.sum(varray[0], axis=0))
        else:
            use_areas = np.isfinite(varray[0])
        # print(use_areas.shape)
        regrid = xesmf.Regridder(varray, self.pressure_field, method=method)
        # Use the regridder
        outfield = regrid(varray.values)
        # Figure out the time coordinates
        if self.timecoord not in list(varray.coords):
            timecoord = "dayhour"
        else:
            timecoord = self.timecoord
        if vcoord is not None:
            outarray = xarray.DataArray(
                outfield,
                dims=(timecoord, vcoord, "lat", "lon"),
                coords={
                    timecoord: varray[timecoord].values,
                    vcoord: varray[vcoord].values,
                    "lat": self.pressure_field["lat"].values,
                    "lon": self.pressure_field["lon"].values,
                },
                attrs=varray.attrs,
                name=varray.name,
            )
        else:
            outarray = xarray.DataArray(
                outfield,
                dims=(timecoord, "lat", "lon"),
                coords={
                    timecoord: varray[timecoord].values,
                    "lat": self.pressure_field["lat"].values,
                    "lon": self.pressure_field["lon"].values,
                },
                attrs=varray.attrs,
                name=varray.name,
            )
        # Clean up the weight file generated by esmf
        regrid.clean_weight_file()
        return outarray

    def backfill_array(self, varray, nans_vary=False):
        """
        Given a 4d array with nans, backfill with nearest non-nan neighbors in space
        Useful for areas where coastline data does not match up and for some
        "underground" interpolation

        nans_vary => Set to True if nan values could be different for different levels and times
        """
        self.logger.info("Backfilling NAN values with nearest neighbors")
        # print("This may take some time...")
        nt, nz, ny, nx = varray.shape

        if nans_vary:
            # Need to recompute mask for each time and level (much slower)
            for t in range(nt):
                for z in range(nz):
                    raw = np.ma.masked_where(
                        np.isnan(varray.values[t, z]), varray.values[t, z]
                    )
                    y, x = np.mgrid[0:ny, 0:nx]
                    xygood = np.array((y[~raw.mask], x[~raw.mask])).T
                    xybad = np.array((y[raw.mask], x[raw.mask])).T
                    the_nearest = KDTree(xygood).query(xybad)[1]
                    thisarr = np.ma.masked_where(
                        np.isnan(varray[t, z].values), varray[t, z].values
                    )
                    thisarr[thisarr.mask] = thisarr[~thisarr.mask][the_nearest]
                    varray[t, z] = thisarr[:]

        else:
            # One mask for all times/levels
            raw = np.ma.masked_where(np.isnan(varray.values[0, 0]), varray.values[0, 0])
            y, x = np.mgrid[0:ny, 0:nx]
            xygood = np.array((y[~raw.mask], x[~raw.mask])).T
            xybad = np.array((y[raw.mask], x[raw.mask])).T
            the_nearest = KDTree(xygood).query(xybad)[1]
            for t in range(nt):
                for z in range(nz):
                    thisarr = np.ma.masked_where(
                        np.isnan(varray[t, z].values), varray[t, z].values
                    )
                    thisarr[thisarr.mask] = thisarr[~thisarr.mask][the_nearest]
                    varray[t, z] = thisarr[:]
        return varray
