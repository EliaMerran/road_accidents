EPS = 30
MIN_SAMPLES = 5
MIN_CLUSTER_SIZE = 5
START_YEAR = 2008
END_YEAR = 2019
TRAIN_INTERVAL = 3
END_TRAIN_YEAR = END_YEAR - TRAIN_INTERVAL + 1
NUM_CITIES = 20


road_type_dict = {
    1: 'Urban at Intersection',
    2: 'Urban Not at Intersection',
    3: 'Non-Urban at Intersection',
    4: 'Non-Urban Not at Intersection'
}
month_dict = {
    1: 'January',
    2: 'February',
    3: ' March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}
day_in_month_dict = {i: str(i) for i in range(1, 32)}
day_night_dict = {
    1: 'Day',
    5: 'Night'
}
day_in_week_dict = {
    1: 'Sunday',
    2: 'Monday',
    3: ' Tuesday',
    4: 'Wednesday',
    5: 'Thursday',
    6: 'Friday',
    7: 'Saturday'
}
traffic_light_dict = {
    1: 'Traffic Light',
    0: 'No Traffic Light',
    9: 'Unknown'
}
accident_type_dict = {
    1: 'Pedestrian Injury',
    2: 'Frontal Collision with Side',
    3: 'Frontal Collision with Rear-End Collision',
    4: 'Side Impact Collision',
    5: 'Head-on Collision',
    6: 'Collision with Parked Vehicle',
    7: 'Collision with Parked Car',
    8: 'Collision with Stationary Object',
    9: 'Exiting or Entering Roadway',
    10: 'Rollover',
    11: 'Overturning',
    12: 'Passenger Injury Inside Vehicle',
    13: 'Falling from Moving Vehicle',
    14: 'Fire',
    15: 'Unknown',
    17: 'Rear-End Collision with Front',
    18: 'Rear-End Collision with Side',
    19: 'Collision with Animal',
    20: 'Cargo Load Impact'
}
road_shape_dict = {
    1: 'Entering Interchange',
    2: 'Exiting Interchange',
    3: 'Parking/Fuel Station',
    4: 'Steep Slope',
    5: 'Sharp Curve',
    6: 'On Bridge or Tunnel',
    7: 'Railroad Crossing',
    8: 'Straight Road/Intersection',
    9: 'Unknown',
    10: 'Bus Stop'
}
single_lane_dict = {
    1: 'Single Lane',
    2: 'Dual Lane with Continuous Divider',
    3: 'Dual Lane without Continuous Divider',
    4: 'Other',
    9: 'Unknown'
}
multi_lane_dict = {
    1: 'Colored Separation',
    2: 'Separation with Safety Barrier',
    3: 'Constructed Separation without Barrier',
    4: 'Unconstructed Separation',
    5: 'Unknown'
}
speed_limit_dict = {
    0: 'Unknown',
    1: 'Up to 50 km/h',
    2: '60 km/h',
    3: '70 km/h',
    4: '80 km/h',
    5: '90 km/h',
    6: '100 km/h',
    7: '110 km/h'
}
defect_type_dict = {
    0: 'Unknown',
    1: 'No Defect',
    2: 'Poor Shoulders',
    3: 'Deteriorated Road Surface',
    4: 'Poor Shoulders and Deteriorated Road Surface'
}
road_width_dict = {
    0: 'Unknown',
    1: 'Up to 5 meters',
    2: '5 to 7 meters',
    3: '7 to 10.5 meters',
    4: '10.5 to 14 meters',
    5: 'More than 14 meters'
}
marking_defect_dict = {
    1: 'Marking Defective/Missing',
    2: 'Reflective Marking Defective/Missing',
    3: 'No Defect',
    4: 'Reflective Marking Not Required',
    5: 'Unknown'
}
lighting_condition_dict = {
    1: 'Normal Daylight',
    2: 'Limited Visibility Due to Weather (Smoke, Haze)',
    3: 'Active Night Lighting',
    4: 'Existing Lighting Malfunctioning/Not Working',
    5: 'No Night Lighting',
    6: 'Night, Unknown',
    7: 'Night, Proper Lighting with Limited Visibility',
    8: 'Night, Faulty Lighting with Limited Visibility',
    9: 'Night, No Lighting with Limited Visibility',
    10: 'Dusk',
    11: 'Daytime, Unknown'
}
weather_condition_dict = {
    1: 'Clear',
    2: 'Rainy',
    3: 'Hazy',
    4: 'Foggy',
    5: 'Other',
    9: 'Unknown'
}
road_condition_dict = {
    1: 'Dry',
    2: 'Wet from Water',
    3: 'Covered with Fuel',
    4: 'Covered with Dust',
    5: 'Sand or Debris on the Road',
    6: 'Other',
    9: 'Unknown'
}

ATTRIBUTES_DICT = {
    'SUG_DEREH': road_type_dict,
    'HODESH_TEUNA': month_dict,
    'YOM_BE_HODESH': day_in_month_dict,
    'YOM_LAYLA': day_night_dict,
    'YOM_BASHAVUA': day_in_week_dict,
    'RAMZOR': traffic_light_dict,
    'SUG_TEUNA': accident_type_dict,
    'ZURAT_DEREH': road_shape_dict,
    'HAD_MASLUL': single_lane_dict,
    'RAV_MASLUL': multi_lane_dict,
    'MEHIRUT_MUTERET': speed_limit_dict,
    'TKINUT': defect_type_dict,
    'ROHAV': road_width_dict,
    'SIMUN_TIMRUR': marking_defect_dict,
    'TEURA': lighting_condition_dict,
    'MEZEG_AVIR': weather_condition_dict,
    'PNE_KVISH': road_condition_dict}

