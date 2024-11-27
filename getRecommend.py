import pandas as pd
import ast
import json
import numpy as np
import os
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

from collaborative_filtering import collaborativefiltering_recommendations

# 1. 데이터 불러오기 및 전처리

# edges.csv와 routes.csv 파일 불러오기
edges = pd.read_csv('edges.csv')
routes = pd.read_csv('routes.csv')

# routes.csv의 열 이름 확인 및 'route_id'로 통일
print("routes.csv의 열 이름:", routes.columns.tolist())

if 'id' in routes.columns:
    routes.rename(columns={'id': 'route_id'}, inplace=True)
elif 'route_id' not in routes.columns:
    raise KeyError("routes.csv 파일에 'route_id' 또는 'id' 열이 존재하지 않습니다.")

# edges.csv의 필수 열 확인
required_edge_columns = {'id', 'source_lat', 'source_long', 'slope', 'is_uphill', 'nearby_features'}
if not required_edge_columns.issubset(edges.columns):
    missing = required_edge_columns - set(edges.columns)
    raise KeyError(f"edges.csv 파일에 필요한 열이 없습니다: {missing}")

# edges 데이터 전처리
edges['nearby_features'] = edges['nearby_features'].fillna('none')

def extract_features(features):
    if isinstance(features, str) and features.lower() == 'none':
        return []
    elif isinstance(features, str):
        return [feature.strip() for feature in features.split(',')]
    else:
        return []

edges['features_list'] = edges['nearby_features'].apply(extract_features)

# routes 데이터 전처리
routes['edge_ids'] = routes['ids'].apply(ast.literal_eval)

def extract_route_tags(tag_str):
    try:
        tag_list = ast.literal_eval(tag_str)
        features = []
        for tag in tag_list:
            feature = tag.split(':')[0].strip('"').strip()
            features.append(feature)
        return features
    except (ValueError, SyntaxError):
        return []

routes['tags_list'] = routes['tag'].apply(extract_route_tags)

# 루트의 모든 특징 합치기
routes['all_features'] = routes['tags_list'] + routes['preferred_feature'].fillna('').apply(lambda x: [x] if x else [])
routes['features_str'] = routes['all_features'].apply(lambda x: ' '.join(x))

# 시작점 좌표 추출 (edges 데이터에 'source_lat', 'source_long'가 있다고 가정)
def get_route_start_point(edge_ids):
    if not edge_ids:
        return (None, None)
    first_edge_id = edge_ids[0]
    edge = edges[edges['id'] == first_edge_id]
    if not edge.empty:
        start_lat = edge['source_lat'].values[0]
        start_lon = edge['source_long'].values[0]
        return (start_lat, start_lon)
    else:
        return (None, None)

routes['start_point'] = routes['edge_ids'].apply(get_route_start_point)

# 시작점 좌표 검증
print("start_point 예시:")
print(routes['start_point'].head())

# 4. 난이도와 거리 스케일링
scaler = MinMaxScaler()
routes[['scaled_distance', 'scaled_difficulty']] = scaler.fit_transform(routes[['total_distance_km', 'difficulty_score']])

# 5. 텍스트 벡터화
vectorizer = TfidfVectorizer()
text_feature_vectors = vectorizer.fit_transform(routes['features_str'])

# 6. 스케일된 수치 특징을 희소 행렬로 변환
scaled_features = routes[['scaled_distance', 'scaled_difficulty']].values
scaled_feature_sparse = csr_matrix(scaled_features)

# 7. 텍스트 벡터와 스케일된 수치 특징 결합
combined_feature_vectors = hstack([text_feature_vectors, scaled_feature_sparse])

# 2. 사용자 데이터 불러오기 또는 생성
users_file = 'user_profiles.json'  # CSV 대신 JSON 사용 권장

if os.path.exists(users_file):
    with open(users_file, 'r', encoding='utf-8') as f:
        users_data = json.load(f)
else:
    users_data = {}

def convert_np_types(obj):
    """
    재귀적으로 딕셔너리나 리스트를 순회하며 numpy 데이터 타입을 표준 Python 타입으로 변환합니다.
    """
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(elem) for elem in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def save_user_profile(user_info):
    global users_data
    users_data[user_info['user_id']] = {
        'height': user_info['height'],
        'weight': user_info['weight'],
        'gender': user_info['gender'],
        'experience_level': user_info['experience_level'],
        'running_records': user_info['running_records']
    }
    
    # JSON 직렬화 가능한 타입으로 변환
    serializable_data = convert_np_types(users_data)
    
    with open(users_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=4)
    # 디버깅 출력
    print(f"[DEBUG] 사용자 '{user_info['user_id']}' 프로파일이 저장되었습니다.")

def get_user_profile(user_id):
    global users_data
    is_new_user = False
    if user_id in users_data:
        # 기존 사용자 로드
        user = users_data[user_id]
        user_info = {
            'user_id': user_id,
            'height': user.get('height'),
            'weight': user.get('weight'),
            'gender': user.get('gender'),
            'experience_level': user.get('experience_level'),
            'running_records': user.get('running_records', []),
            'user_profile': None,
            'is_new_user': False
        }
    else:
        # 신규 사용자 생성
        is_new_user = True
        experience_level = input("운동 경력 수준을 입력하세요 (beginner/intermediate/advanced): ").strip().lower()
        while experience_level not in ['beginner', 'intermediate', 'advanced']:
            experience_level = input("잘못된 입력입니다. 다시 입력하세요 (beginner/intermediate/advanced): ").strip().lower()

        user_info = {
            'user_id': user_id,
            'height': None,
            'weight': None,
            'gender': None,
            'experience_level': experience_level,
            'running_records': [],
            'user_profile': None,
            'is_new_user': True
        }
        # 사용자 데이터에 추가
        users_data[user_id] = {
            'height': user_info['height'],
            'weight': user_info['weight'],
            'gender': user_info['gender'],
            'experience_level': experience_level,
            'running_records': []
        }
        save_user_profile(user_info)
    user_info['is_new_user'] = is_new_user
    return user_info

# 사용자 ID 입력 받기
user_id = input("사용자 ID를 입력하세요: ").strip()
user_info = get_user_profile(user_id)
if user_info['is_new_user']:
    try:
        height = float(input("키(cm)를 입력하세요: ").strip())
    except ValueError:
        height = None
        print("잘못된 입력입니다. 키를 입력하지 않습니다.")
    try:
        weight = float(input("몸무게(kg)를 입력하세요: ").strip())
    except ValueError:
        weight = None
        print("잘못된 입력입니다. 몸무게를 입력하지 않습니다.")
    gender = input("성별을 입력하세요 (male/female): ").strip().lower()
    while gender not in ['male', 'female']:
        gender = input("잘못된 입력입니다. 다시 입력하세요 (male/female): ").strip().lower()
    # 사용자 정보에 추가
    user_info['height'] = height
    user_info['weight'] = weight
    user_info['gender'] = gender
    save_user_profile(user_info)

# 레벨별 거리 및 난이도 범위 정의
LEVEL_SETTINGS = {
    'beginner': {
        'min_distance': 2,
        'max_distance': 4,
        'avg_difficulty': 0.5
    },
    'intermediate': {
        'min_distance': 4,
        'max_distance': 7,
        'avg_difficulty': 3
    },
    'advanced': {
        'min_distance': 7,
        'max_distance': 10,
        'avg_difficulty': 5
    }
}

# 3. 사용자 운동 경력에 따른 추천 조건 설정 (레벨별 거리 범위 적용)
def get_distance_difficulty_range(user_info, condition):
    """
    사용자 운동 경력과 컨디션에 따라 추천할 거리와 난이도 범위를 반환합니다.
    """
    # 사용자 운동 경력 수준에 따른 거리 및 난이도 범위 가져오기
    experience_level = user_info['experience_level']
    level_settings = LEVEL_SETTINGS.get(experience_level, LEVEL_SETTINGS['intermediate'])
    min_distance = level_settings['min_distance']
    max_distance = level_settings['max_distance']
    avg_difficulty = level_settings['avg_difficulty']
    
    # 사용자 최근 러닝 기록에서 평균 거리 계산 (최근 3개)
    running_records = user_info['running_records']
    recent_records = running_records[-3:] if len(running_records) >= 3 else running_records
    if recent_records:
        total_distance = sum(
            routes[routes['route_id'] == record['route_id']]['total_distance_km'].values[0]
            for record in recent_records
            if not routes[routes['route_id'] == record['route_id']].empty
        )
        avg_distance = total_distance / len(recent_records)
    else:
        avg_distance = min_distance

    # 컨디션에 따른 거리와 난이도 조절 (30% 범위)
    condition_factor = 0.3
    if condition == 'good':
        adjusted_distance = avg_distance * (1 + condition_factor)
        adjusted_difficulty = avg_difficulty * (1 + condition_factor)
        # 'good'일 때는 max_distance를 초과하지 않도록
        adjusted_distance = min(adjusted_distance, max_distance)
        adjusted_difficulty = min(adjusted_difficulty, 10)  # 예시로 최대 난이도 10으로 설정
    elif condition == 'normal':
        adjusted_distance = avg_distance
        adjusted_difficulty = avg_difficulty
        # 'normal'일 때는 min_distance와 max_distance 사이로 유지
        adjusted_distance = max(min_distance, min(adjusted_distance, max_distance))
        adjusted_difficulty = min(adjusted_difficulty, 10)
    elif condition == 'bad':
        adjusted_distance = avg_distance * (1 - condition_factor)
        adjusted_difficulty = avg_difficulty * (1 - condition_factor)
        # 'bad'일 때는 min_distance를 적용하지 않음, 더 줄일 수 있음
    else:
        adjusted_distance = avg_distance
        adjusted_difficulty = avg_difficulty
        # 다른 조건일 때는 min/max 적용
        adjusted_distance = max(min_distance, min(adjusted_distance, max_distance))
        adjusted_difficulty = min(adjusted_difficulty, 10)
    
    return adjusted_distance, adjusted_difficulty


# 사용자 컨디션 입력 받기
condition = input("오늘의 컨디션은 어떠신가요? (good/normal/bad): ").strip().lower()
while condition not in ['good', 'normal', 'bad']:
    condition = input("잘못된 입력입니다. 다시 입력하세요 (good/normal/bad): ").strip().lower()

# 사용자 위치 입력 받기 (저장하지 않음)
def get_user_location():
    """
    사용자가 추천을 받을 때마다 위치를 새로 입력할 수 있도록 합니다.
    위치 정보를 저장하지 않습니다.
    """
    while True:
        new_location = input("현재 위치를 입력하세요 (위도,경도): ").strip()
        try:
            lat, lon = map(float, new_location.split(','))
            print(f"현재 위치가 ({lat}, {lon})으로 설정되었습니다.")
            return (lat, lon)
        except ValueError:
            print("잘못된 형식입니다. 위도와 경도를 쉼표로 구분하여 입력하세요 (예: 37.5665,126.9780).")

# 사용자 위치 입력 받기 (저장하지 않음)
user_location = get_user_location()

# 사용자 프로파일 생성 함수 개선
def create_user_profile_vector(user_info, vectorizer, scaler, preferred_feature):
    """
    사용자의 선호도와 과거 러닝 기록을 기반으로 사용자 프로파일 벡터를 생성합니다.
    """
    running_records = user_info['running_records']
    if not running_records:
        # 사용자 기록이 없을 경우 기본 프로파일 반환
        feature_str = preferred_feature if preferred_feature != 'any' else ''
        text_vector = vectorizer.transform([feature_str])
        numerical_vector = csr_matrix([[0, 0]])
        user_profile_vector = hstack([text_vector, numerical_vector])
        return user_profile_vector

    # 선호도에 따라 러닝 기록 필터링
    if preferred_feature != 'any':
        filtered_records = [record for record in running_records if record.get('preferences') == preferred_feature]
        if not filtered_records:
            # 해당 선호도의 기록이 없을 경우 기본 프로파일 반환
            feature_str = preferred_feature
            text_vector = vectorizer.transform([feature_str])
            numerical_vector = csr_matrix([[0, 0]])
            user_profile_vector = hstack([text_vector, numerical_vector])
            return user_profile_vector
    else:
        filtered_records = running_records

    # 각 기록에 대해 가중치를 적용하여 특징 추출
    total_rating = sum(record['rating'] for record in filtered_records)
    if total_rating == 0:
        total_rating = 1  # ZeroDivisionError 방지

    # 가중 평균 계산을 위한 준비
    weighted_features = []
    weighted_distances = []
    weighted_difficulties = []

    for record in filtered_records:
        route = routes[routes['route_id'] == record['route_id']]
        if not route.empty:
            weight = record['rating'] / total_rating
            features = route['features_str'].values[0]
            weighted_features.extend([features] * record['rating'])  # 텍스트 특징은 빈도에 가중치 적용
            distance = route['total_distance_km'].values[0]
            difficulty = route['difficulty_score'].values[0]
            weighted_distances.append(distance * weight)
            weighted_difficulties.append(difficulty * weight)

    # 텍스트 벡터화
    combined_features_str = ' '.join(weighted_features)
    text_vector = vectorizer.transform([combined_features_str])

    # 수치 특징 계산
    avg_distance = sum(weighted_distances)
    avg_difficulty = sum(weighted_difficulties)

    # 스케일링
    scaled_features = scaler.transform([[avg_distance, avg_difficulty]])
    numerical_vector = csr_matrix(scaled_features)

    # 텍스트와 수치 특징 결합
    user_profile_vector = hstack([text_vector, numerical_vector])

    return user_profile_vector

# Haversine 거리 계산 함수 (미터 단위로 반환)
def haversine_distance(coord1, coord2):
    """
    두 지점 간의 거리(m)를 계산합니다.
    coord1, coord2: (latitude, longitude)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    if None in coord1 or None in coord2:
        return float('inf')  # 무한대로 설정하여 필터링
    R = 6371000.0  # 지구 반경 (m)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    meters = R * c  # 거리(m)
    return meters

# 두 루트의 시작점 간 거리가 최소 30m 이상인지 확인하는 함수
def are_start_points_far_enough(new_route_start, selected_routes):
    """
    새로운 루트의 시작점과 이미 선택된 루트들의 시작점 간 거리가 최소 30m 이상인지 확인합니다.
    """
    for route in selected_routes:
        distance = haversine_distance(new_route_start, route['start_point'])
        if distance < 30:
            return False
    return True

# filter_routes_by_location 함수 수정 (위치 정보 저장 제거)
def filter_routes_by_location(routes_df, user_location, edges, radius=0.5):
    """
    사용자 위치에서 반경 radius km 내에 시작 노드가 있는 루트들을 필터링합니다.
    """
    # 시작점의 위도와 경도를 추출
    start_lats = routes_df['start_point'].apply(lambda x: x[0] if x[0] is not None else np.nan)
    start_lons = routes_df['start_point'].apply(lambda x: x[1] if x[1] is not None else np.nan)
    
    # NaN이 있는 경우 필터링
    valid_indices = (~start_lats.isna()) & (~start_lons.isna())
    routes_df = routes_df[valid_indices].copy()
    
    # 사용자 위치와 각 시작점 간의 거리 계산
    distances = routes_df.apply(lambda row: haversine_distance(user_location, row['start_point']), axis=1)
    routes_df['distance_to_user'] = distances
    
    # 반경 내의 루트만 필터링
    filtered_routes = routes_df[routes_df['distance_to_user'] <= radius * 1000].copy()  # radius를 미터로 변환
    
    print(f"[DEBUG] 필터링 전 루트 개수: {len(routes_df)}, 필터링 후 루트 개수: {len(filtered_routes)}")
    
    return filtered_routes

# 6. 루트 추천 함수 개선 (Edge Preference Score 제거 및 추가 요구사항 반영)
def recommend_routes(user_info, vectorizer, scaler, combined_feature_vectors, routes_df, condition, top_n=3):
    """
    사용자 시나리오에 따라 루트를 추천합니다.
    """
    # 오늘의 선호도 입력
    preferred_feature = input("\n오늘 선호하는 운동 코스를 입력하세요 (예: river, mountain, park, any): ").strip().lower()
    while preferred_feature not in ['river', 'mountain', 'park', 'any']:
        preferred_feature = input("잘못된 입력입니다. 다시 입력하세요 (river, mountain, park, any): ").strip().lower()
    
    # 사용자 프로파일 벡터 생성
    user_profile_vector = create_user_profile_vector(user_info, vectorizer, scaler, preferred_feature)
    
    # 컨디션에 따른 거리와 난이도 조절 (30% 범위)
    adjusted_distance, adjusted_difficulty = get_distance_difficulty_range(user_info, condition)
    
    # 거리와 난이도 필터링
    routes_df_filtered = routes_df[
        (routes_df['total_distance_km'] >= adjusted_distance * 0.9) &  # ±10%
        (routes_df['total_distance_km'] <= adjusted_distance * 1.1) &
        (routes_df['difficulty_score'] <= adjusted_difficulty * 1.1)
    ].copy()
    
    print(f"[DEBUG] 거리 및 난이도 필터링 후 루트 개수: {len(routes_df_filtered)}")
    
    # 사용자 위치 필터링 (반경 500m 내)
    routes_df_filtered = filter_routes_by_location(routes_df_filtered, user_location, edges, radius=0.5)
    
    # 이미 평가한 루트 제외 (선택사항: 필요 시 제거)
    evaluated_route_ids = [record['route_id'] for record in user_info['running_records']]
    routes_df_filtered = routes_df_filtered[~routes_df_filtered['route_id'].isin(evaluated_route_ids)]
    
    print(f"[DEBUG] 이미 평가한 루트 제외 후 루트 개수: {len(routes_df_filtered)}")
    
    if routes_df_filtered.empty:
        print("[DEBUG] 필터링된 루트가 없습니다.")
        # 이전에 뛰었던 해당 선호도의 루트 중 rating이 가장 높은 루트 추천
        if preferred_feature != 'any':
            filtered_past_runs = [record for record in user_info['running_records'] if record.get('preferences') == preferred_feature]
        else:
            filtered_past_runs = user_info['running_records']
        
        if not filtered_past_runs:
            print("이전 러닝 기록이 없어 추천할 루트가 없습니다.")
            return routes_df_filtered, preferred_feature
        
        # rating이 가장 높은 루트 선택 (여러 개일 경우 첫 번째 선택)
        sorted_past_runs = sorted(filtered_past_runs, key=lambda x: x['rating'], reverse=True)
        top_past_run = sorted_past_runs[0]
        route_id = top_past_run['route_id']
        route = routes[routes['route_id'] == route_id]
        if not route.empty:
            print(f"[DEBUG] 이전에 뛰었던 루트 중 rating이 가장 높은 루트 ID: {route_id}")
            # 반환 형식에 맞추기 위해 DataFrame 생성
            recommendation = pd.DataFrame([{
                'route_id': route_id,
                'similarity': 1.0,  # 유사도는 최대값으로 설정
                'total_distance_km': route['total_distance_km'].values[0],
                'difficulty_score': route['difficulty_score'].values[0]
            }])
            return recommendation, preferred_feature
        else:
            print("선택한 루트의 정보가 없습니다.")
            return routes_df_filtered, preferred_feature
    
    # 필터된 루트의 특징 벡터 추출
    route_vectors_filtered = combined_feature_vectors[routes_df_filtered.index]
    
    # 사용자 프로파일과 필터된 루트 간의 콘텐츠 기반 유사도 계산
    similarities = cosine_similarity(user_profile_vector, route_vectors_filtered).flatten()
    routes_df_filtered['similarity'] = similarities
    
    # 유사도 기준으로 정렬
    recommendations = routes_df_filtered.sort_values(by='similarity', ascending=False)

    # 시작점이 다른 루트 선택 및 시작점 간 최소 30m 거리 유지
    selected_recommendations = []
    for idx, row in recommendations.iterrows():
        if len(selected_recommendations) >= top_n:
            break
        # 현재 루트의 시작점
        current_start = row['start_point']
        # 이미 선택된 루트들과의 거리 검사
        if are_start_points_far_enough(current_start, selected_recommendations):
            selected_recommendations.append(row)
    
    if selected_recommendations:
        recommendations = pd.DataFrame(selected_recommendations)
    else:
        recommendations = pd.DataFrame()
    if users_data[user_id]['running_records']:
        hybrid_recommendation = collaborativefiltering_recommendations(user_id, recommendations, file_name="CoClustering_tuned", k=3)
        return hybrid_recommendation[['route_id', 'similarity', 'total_distance_km', 'difficulty_score', 'total_similarity_score']], preferred_feature
    else:
        return recommendations[['route_id', 'similarity', 'total_distance_km', 'difficulty_score', 'similarity']], preferred_feature

# 추천 루트 평가 함수 수정 (평가를 1-5 점수로 받음)
def evaluate_routes(recommendations, user_info, preferred_feature):
    """
    추천된 루트 중 사용자가 하나를 선택하고 평가받습니다.
    선택한 루트에 대해서만 평가를 받고, 데이터베이스에 반영합니다.
    """
    if recommendations.empty:
        return []
    
    # 추천된 루트 목록 출력
    print("\n=== 추천된 루트 목록 ===")
    for idx, row in recommendations.iterrows():
        print(f"{idx + 1}. 루트 ID: {row['route_id']}, 거리: {row['total_distance_km']}km, 난이도: {row['difficulty_score']}")
    
    # 사용자가 하나의 루트 선택
    selected_idx = input("\n달리고 싶은 루트의 번호를 선택하세요 (예: 1): ").strip()
    if not selected_idx.isdigit():
        print("잘못된 입력입니다.")
        return []
    
    selected_idx = int(selected_idx) - 1
    if selected_idx not in range(len(recommendations)):
        print("선택한 번호가 유효하지 않습니다.")
        return []
    
    selected_route = recommendations.iloc[selected_idx]
    route_id = selected_route['route_id']
    print(f"\n선택한 루트 ID: {route_id}")
    print(f"루트 정보: 거리 {selected_route['total_distance_km']}km, 난이도 {selected_route['difficulty_score']}")
    
    # 추가 정보 계산
    edge_ids = routes[routes['route_id'] == route_id]['edge_ids'].values[0]
    route_edges = edges[edges['id'].isin(edge_ids)].copy()
    if not route_edges.empty:
        average_slope = route_edges['slope'].mean()
        uphill_count = route_edges['is_uphill'].sum()
        print(f"평균 경사도: {average_slope:.4f}")
        print(f"오르막길 개수: {uphill_count}")
    else:
        average_slope = 0
        uphill_count = 0
        print("엣지 정보가 없습니다.")
    
    # 달리기 시간 입력
    try:
        time_taken = float(input("이 루트를 달리는 데 걸린 시간을 입력하세요(분 단위): ").strip())
    except ValueError:
        print("잘못된 입력입니다. 시간을 0으로 설정합니다.")
        time_taken = 0.0
    
    # 루트 평가 입력 (1-5 점수)
    try:
        rating = int(input("이 루트를 1부터 5까지의 점수로 평가하세요 (1: 최악, 5: 최고): ").strip())
        while rating < 1 or rating > 5:
            rating = int(input("잘못된 입력입니다. 1부터 5까지의 점수를 입력하세요: ").strip())
    except ValueError:
        print("잘못된 입력입니다. 점수를 3으로 설정합니다.")
        rating = 3
    
    # 러닝 기록 저장
    running_record = {
        'route_id': route_id,
        'time_taken': time_taken,
        'preferences': preferred_feature,
        'rating': rating
    }
    user_info['running_records'].append(running_record)
    
    # 사용자 정보 업데이트
    save_user_profile(user_info)
    
    return [route_id]

# 사용자 레벨 업데이트 함수 수정
def update_user_experience_level(user_data, routes_df):
    """
    사용자의 러닝 기록을 기반으로 운동 경력 수준을 업데이트합니다.
    """
    running_records = user_data['running_records']
    if not running_records or len(running_records) < 3:
        return  # 기록이 3개 미만이면 레벨 변경하지 않음

    # 3번째 기록부터 사용
    recent_records = running_records[2:]  # 인덱스 2부터 끝까지

    # 러닝 기록에서 distance와 difficulty_score를 routes_df를 통해 조회
    distances = []
    times = []
    paces = []
    for record in recent_records:
        route = routes_df[routes_df['route_id'] == record['route_id']]
        if not route.empty:
            distance = route['total_distance_km'].values[0]
            time = record['time_taken']
            pace = time / distance if distance > 0 else 0  # 분/km
            distances.append(distance)
            times.append(time)
            paces.append(pace)

    if not distances or not paces:
        return

    # 거리 가중치를 적용한 평균 페이스 계산
    total_distance = sum(distances)
    weighted_pace_sum = sum(pace * distance for pace, distance in zip(paces, distances))
    average_pace = weighted_pace_sum / total_distance if total_distance > 0 else 0

    # 평균 거리 계산
    average_distance = total_distance / len(distances)

    # 레벨 업그레이드 조건
    if user_data['experience_level'] == 'beginner':
        if average_pace < 7 and average_distance >= 3:
            user_data['experience_level'] = 'intermediate'
            print("축하합니다! 'intermediate' 레벨로 업그레이드되었습니다.")
    elif user_data['experience_level'] == 'intermediate':
        if average_pace < 6 and average_distance >= 5:
            user_data['experience_level'] = 'advanced'
            print("축하합니다! 'advanced' 레벨로 업그레이드되었습니다.")
    # 레벨 다운 가능성 추가
    elif user_data['experience_level'] == 'advanced':
        if average_pace > 7 or average_distance < 5:
            user_data['experience_level'] = 'intermediate'
            print("'intermediate' 레벨로 다운그레이드되었습니다.")

    save_user_profile(user_data)
    
# 사용자 레벨 업데이트
update_user_experience_level(user_info, routes)

def save_recommended_routes_to_csv(recommendations, filename='recommended_routes.csv'):
    """
    추천된 루트의 ID를 CSV 파일로 저장합니다.
    """
    recommendations[['route_id']].to_csv(filename, index=False)
    print(f"\n추천된 루트 ID가 '{filename}' 파일에 저장되었습니다.")

# 11. 업데이트된 프로파일을 통한 재추천
updated_recommendations, preferred_feature = recommend_routes(user_info, vectorizer, scaler, combined_feature_vectors, routes, condition)
if not updated_recommendations.empty:
    print("\n=== 추천된 루트 목록 ===")
    for idx, row in updated_recommendations.iterrows():
        print(f"{idx + 1}. 루트 ID: {row['route_id']}, 합산 유사도 점수: {row['total_similarity_score']:.4f}, "
              f"거리: {row['total_distance_km']}km, 난이도: {row['difficulty_score']}")
    # 추천된 루트 ID를 CSV로 저장
    save_recommended_routes_to_csv(updated_recommendations)
else:
    print("\n추천할 루트가 없습니다.")

# 12. 재추천된 루트에 대해 선택 및 평가 받기
if not updated_recommendations.empty:
    selected_route = evaluate_routes(updated_recommendations, user_info, preferred_feature)
