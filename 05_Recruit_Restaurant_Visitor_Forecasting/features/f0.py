visit_num_vars = ['dow', 'year', 'month', 'doy', 'day', 'woy', 'is_month_end', 'date_int']

reserve_num_vars = [
    'reserve_ppl_count_air', 'reserve_tot_count_air', 'avg_reserve_hr_day_air',
    'max_reserve_hr_air', 'min_reserve_hr_air', 'mean_reserve_hr_air', 'min_reserve_dy_air',
    'mean_reserve_dy_air', 'max_reserve_dy_air', 'reserve_ppl_count_hpg', 'reserve_tot_count_hpg',
    'avg_reserve_hr_day_hpg', 'max_reserve_hr_hpg', 'min_reserve_hr_hpg', 'mean_reserve_hr_hpg',
    'min_reserve_dy_hpg', 'mean_reserve_dy_hpg', 'max_reserve_dy_hpg'
]

store_num_vars = [
    'latitude_air', 'longitude_air', 'air_stores_on_same_addr', 'air_stores_lv1',
    'air_stores_lv2', 'air_stores_lv3', 'mean_lat_air_lv1', 'max_lat_air_lv1',
    'min_lat_air_lv1', 'mean_lon_air_lv1', 'max_lon_air_lv1', 'min_lon_air_lv1',
    'mean_lat_air_lv2',  'max_lat_air_lv2', 'min_lat_air_lv2', 'mean_lon_air_lv2',
    'max_lon_air_lv2',   'min_lon_air_lv2', 'air_genre_count', 'air_genre_count_lv1',
    'air_genre_count_lv2',   'air_genre_count_lv3', 'latitude_hpg', 'longitude_hpg',
    'hpg_stores_on_same_addr', 'hpg_stores_lv1', 'hpg_stores_lv2', 'hpg_stores_lv3',
    'mean_lat_hpg_lv1', 'max_lat_hpg_lv1', 'min_lat_hpg_lv1', 'mean_lon_hpg_lv1',
    'max_lon_hpg_lv1', 'min_lon_hpg_lv1', 'mean_lat_hpg_lv2', 'max_lat_hpg_lv2',
    'min_lat_hpg_lv2', 'mean_lon_hpg_lv2', 'max_lon_hpg_lv2', 'min_lon_hpg_lv2',
    'hpg_genre_count', 'hpg_genre_count_lv1', 'hpg_genre_count_lv2', 'hpg_genre_count_lv3'
    ]
store_cat_vars = [
    'air_genre_name', 'air_lv1', 'air_lv2', 'air_lv3', 'air_lv4',
    'hpg_genre_name', 'hpg_lv1', 'hpg_lv2', 'hpg_lv3'
]

interacts_vars = [
    'reserve_ppl_count', 'reserve_tot_count', 'reserve_ppl_mean', 'lon_plus_lat_air',
    'lat_to_mean_lat_air_lv1', 'lat_to_max_lat_air_lv1', 'lat_to_min_lat_air_lv1',
    'lon_to_mean_lon_air_lv1', 'lon_to_max_lon_air_lv1', 'lon_to_min_lon_air_lv1',
    'lat_to_mean_lat_air_lv2', 'lat_to_max_lat_air_lv2', 'lat_to_min_lat_air_lv2',
    'lon_to_mean_lon_air_lv2', 'lon_to_max_lon_air_lv2', 'lon_to_min_lon_air_lv2',
    'lat_to_mean_lat_hpg_lv1', 'lat_to_max_lat_hpg_lv1', 'lat_to_min_lat_hpg_lv1',
    'lon_to_mean_lon_hpg_lv1', 'lon_to_max_lon_hpg_lv1', 'lon_to_min_lon_hpg_lv1',
    'lat_to_mean_lat_hpg_lv2', 'lat_to_max_lat_hpg_lv2', 'lat_to_min_lat_hpg_lv2',
    'lon_to_mean_lon_hpg_lv2', 'lon_to_max_lon_hpg_lv2', 'lon_to_min_lon_hpg_lv2',
]

hol_mix_vars = ['weight1', 'weight2', 'holiday_flg']

target_agg_vars = ['mean_visitors', 'max_visitors', 'min_visitors', 'median_visitors', 'wmean_visitors']

features = visit_num_vars + reserve_num_vars + store_num_vars + store_cat_vars + interacts_vars + hol_mix_vars + target_agg_vars

def features_set_f0():
    return features