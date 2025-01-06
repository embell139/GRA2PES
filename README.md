<h1>Generating GRA2PES Content for the GHG Center</h1>

Native GRA2PES files are delivered as hourly data in 12-hourly files, two per day, for three days: satdy, sundy, weekdy. These are stored in 
`/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES/<YYYY>/Month0<N>/<Sector>/<day>`
with filenames of the format 
`GRA2PESv1.0_OFFROAD_202101_satdy_00to11Z.nc`.
These files contain TOTAL emissions, across all sectors, for five species: CO2, CO, NOx, SO2, and PM2.5. 

<h2>EnA monthly emission rates</h2>
The EnA environment displays a monthly average of each species' emission rates, summed over all sectors.

To generate, follow these steps:
    1. Run `aggregate_monthly_gra2pes.py`
 
        - `inputs.sectors = ['total']` to sum up all sectors.
        - **Description:** Calculate average monthly emissions rate, regrid to EPSG 4326, convert moles to metric tons where applicable, output to NetCDF.
        - Outputs are stored in `/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES/<YYYY>/Month0<N>`
        - For easiest sync to GHGC S3 bucket, copy outputs to `/data/GRA2PES/for_ghgc/monthly_subset_regrid` directory or similar.
    2. Double check using `double_checks/monthly_aggregates_double_check.p0y` 
        - Prints `Great sum well done!` if the aggregate monthly output matches the validation aggregate to < 0.025% in the mean.

<h2>Urban Dashboard city totals</h2>
- The Urban Dashboard requires a minimized number of sectors, for effective visualization. Steps:
    1. `sectoral_aggregation.py` 
        - `inputs.sum` keywords are the aggregated sector names, and their corresponding lists are the individual sectors to be included in each. 
        - **Description:** Simply add together native hourly data, duplicating the 12-hour, satdy/sundy/mondy output file format.
        - Creates a new sectoral directory in `data/GRA2PES/<year>` and populates the hourly 0to11Z and 12to23Z files with aggregated emissions rates.
    2. Double check using `sector_aggregates_double_check.py`
        - `inputs.sum` should match the same field in `sectoral_aggregation.py`
        - **Description:** Calculates sums using a different method, compares to outputs from step 1.
          Will print `'Check your arithmetic!'` if the sums do not match. 
          Otherwise, `'Great sum well done!'`
    3. Run `aggregate_monthly_gra2pes.py` 
        - Update `inputs.sectors` to include your new aggregated sectors (should be the same as `inputs.sum` from the previous two steps).
        - **Description:** Convert hourly emissions rate to average **monthly** emissions rate, regrid to EPSG 4326, convert moles to metric tons where applicable.
    4. Double check using `quick_total_sectors_check.py` 
        - **Description:** Outputs maps of reported Total emissions, native/aggregated sector emissions, and a map of % diff between reported Total and sum of sectors.
        - This is a visual spot check! Can be used on either native sectors or aggregated sectors.
    4. Run `city_totals.py` 
        - **Description:** Reads in the emissions per month, subsets them for the specified list of GHGC-focused cities, and calculates total emissions for the city for that month.
        - Output is one CSV per city, with chemical species as rows and sectors as columns. It reports sectoral breakdwon in both mass and as a percentage of the total.
        - CSVs get written directly to `/data/GRA2PES/for_ghgc/city_totals`.
    5. Double check using `city_annual_totals_vs_vulcan.py`
        - **Description:** Compares monthly TOTAL, ANNUAL emissions from GRA2PES to those from Vulcan Scope 1/2 in a bar graph, as a sanity check.
        - Note that this only works for whichever time period the two datasets overlap, and that we only have annual totals for Vulcan.
