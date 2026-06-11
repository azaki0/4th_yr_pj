import heapq
import re

WALKING_SPEED_FT_PER_SEC = 2

# 28 pixels = 52.167 ft
NODES = {
    "entrance": {"label": "Main Entrance", "x": 858, "y": 853},
    "entrance_b": {"label": "Entrance B", "x": 1252, "y": 704},
    "main_building": {"label": "Main Building", "x": 856, "y": 650},
    "workshop_right_entrance": {"label": "Workshop Right Entrance", "x": 1098, "y": 666},
    "workshop_left_entrance": {"label": "Workshop Left Entrance", "x": 1006, "y": 666},
    "teacher_dormitory_A": {"label": "Teacher Dormitory A", "x": 1152, "y": 715},
    "teacher_dormitory_B": {"label": "Teacher Dormitory B", "x": 1173, "y": 674},
    "teacher_dormitory_C": {"label": "Teacher Dormitory C", "x": 1153, "y": 583},
    "teacher_dormitory_D": {"label": "Teacher Dormitory D", "x": 939, "y": 236},
    "view_point": {"label": "View Point", "x": 862, "y": 335},
    "boys_dormitory_A": {"label": "Boys Dormitory A", "x": 640, "y": 565},
    "boys_dormitory_B": {"label": "Boys Dormitory B", "x": 536, "y": 568},
    "canteen": {"label": "Canteen", "x": 388, "y": 583},
    "girls_dormitory": {"label": "Girls Dormitory", "x": 290, "y": 490},
    "stadium": {"label": "Stadium", "x": 390, "y": 796},
    "main_entrance_before_roundabout_junction": {"label": "Main Entrance Before Roundabout Junction", "x": 858, "y": 734},
    "roundabout_left_junction": {"label": "Roundabout Left Junction", "x": 820, "y": 700},
    "roundabout_right_junction": {"label": "Roundabout Right Junction", "x": 890, "y": 700},
    "main_building_left_entrance": {"label": "Main Building Left Entrance", "x": 820, "y": 660},
    "main_building_right_entrance": {"label": "Main Building Right Entrance", "x": 890, "y": 660},
    "before_workshop_junction": {"label": "Before Workshop Junction", "x": 1000, "y": 705},
    "after_workshop_junction": {"label": "After Workshop Junction", "x": 1115, "y": 705},
    "before_teacher_dormitory_junction_AB": {"label": "Before Teacher Dormitory Junction AB", "x": 1150, "y": 705},
    "after_teacher_dormitory_junction_AB": {"label": "After Teacher Dormitory Junction AB", "x": 1200, "y": 705},
    "to_viewpoint_node_1": {"label": "To Viewpoint Node 1", "x": 1114, "y": 666},
    "to_viewpoint_node_2": {"label": "To Viewpoint Node 2", "x": 1104, "y": 610},
    "to_viewpoint_node_3": {"label": "To Viewpoint Node 3", "x": 1091, "y": 568},
    "to_viewpoint_node_4": {"label": "To Viewpoint Node 4", "x": 1050, "y": 510},
    "to_viewpoint_node_5": {"label": "To Viewpoint Node 5", "x": 985, "y": 473},
    "to_viewpoint_node_6": {"label": "To Viewpoint Node 6", "x": 950, "y": 447},
    "to_viewpoint_node_7": {"label": "To Viewpoint Node 7", "x": 980, "y": 400},
    "to_viewpoint_node_8": {"label": "To Viewpoint Node 8", "x": 973, "y": 356},
    "to_viewpoint_node_9": {"label": "To Viewpoint Node 9", "x": 930, "y": 300},
    "to_viewpoint_node_10": {"label": "To Viewpoint Node 10", "x": 927, "y": 238},
    "to_viewpoint_node_11": {"label": "To Viewpoint Node 11", "x": 863, "y": 226},
    "to_viewpoint_node_12": {"label": "To Viewpoint Node 12", "x": 845, "y": 270},
    "to_viewpoint_node_13": {"label": "To Viewpoint Node 13", "x": 850, "y": 320},
    "to_dorms_junction_1": {"label": "To Dorms Junction 1", "x": 660, "y": 705},
    "to_dorms_junction_2": {"label": "To Dorms Junction 2", "x": 655, "y": 645},
    "to_dorms_junction_3": {"label": "To Dorms Junction 3", "x": 625, "y": 620},
    "boys_dormitory_junction": {"label": "Boys Dormitory Junction", "x": 590, "y": 620},
    "to_boys_dormitory_A": {"label": "To Boys Dormitory A", "x": 610, "y": 603},
    "to_boys_dormitory_B": {"label": "To Boys Dormitory B", "x": 550, "y": 603},
    "before_cafeteria_junction": {"label": "Before Cafeteria Junction", "x": 412, "y": 620},
    "after_cafeteria_junction": {"label": "After Cafeteria Junction", "x": 368, "y": 620},
    "stadium_entrance": {"label": "Stadium Entrance", "x": 412, "y": 792},
    "girls_dormitory_junction": {"label": "Girls Dormitory Junction", "x": 243, "y": 620},
}

#converted to distance ft
GRAPH = {
    "entrance": {"main_entrance_before_roundabout_junction": 221.71},
    "entrance_b": {"after_teacher_dormitory_junction_AB": 96.9},
    "main_building": {"main_building_left_entrance": 69.61, "main_building_right_entrance": 66.03},
    "workshop_left_entrance": {"before_workshop_junction": 73.52},
    "workshop_right_entrance": {"to_viewpoint_node_1": 29.81},
    "teacher_dormitory_A": {"before_teacher_dormitory_junction_AB": 19.0, "after_teacher_dormitory_junction_AB": 91.35},
    "teacher_dormitory_B": {"before_teacher_dormitory_junction_AB": 71.92, "after_teacher_dormitory_junction_AB": 76.59},
    "teacher_dormitory_C": {"to_viewpoint_node_2": 104.23},
    "teacher_dormitory_D": {"to_viewpoint_node_10": 22.67},
    "view_point": {"to_viewpoint_node_13": 35.79},
    "boys_dormitory_A": {"to_boys_dormitory_A": 90.2},
    "boys_dormitory_B": {"to_boys_dormitory_B": 70.23},
    "canteen": {"before_cafeteria_junction": 82.17, "after_cafeteria_junction": 78.36},
    "girls_dormitory": {"girls_dormitory_junction": 257.55},
    "stadium": {"stadium_entrance": 41.66},
    "main_entrance_before_roundabout_junction": {"entrance": 221.71, "roundabout_left_junction": 95.0, "roundabout_right_junction": 86.99},
    "roundabout_left_junction": {"main_entrance_before_roundabout_junction": 95.0, "main_building_left_entrance": 74.52, "to_dorms_junction_1": 298.24},
    "roundabout_right_junction": {"main_entrance_before_roundabout_junction": 86.99, "main_building_right_entrance": 74.52, "before_workshop_junction": 205.15},
    "main_building_left_entrance": {"roundabout_left_junction": 74.52, "main_building": 69.61},
    "main_building_right_entrance": {"roundabout_right_junction": 74.52, "main_building": 66.03},
    "before_workshop_junction": {"roundabout_right_junction": 205.15, "workshop_left_entrance": 73.52, "after_workshop_junction": 214.26},
    "after_workshop_junction": {"before_workshop_junction": 214.26, "before_teacher_dormitory_junction_AB": 65.21, "to_viewpoint_node_1": 72.69},
    "before_teacher_dormitory_junction_AB": {"after_workshop_junction": 65.21, "teacher_dormitory_A": 19.0, "teacher_dormitory_B": 71.92, "after_teacher_dormitory_junction_AB": 93.16},
    "after_teacher_dormitory_junction_AB": {"teacher_dormitory_A": 91.35, "teacher_dormitory_B": 76.59, "entrance_b": 96.9, "before_teacher_dormitory_junction_AB": 93.16},
    "to_viewpoint_node_1": {"after_workshop_junction": 72.69, "to_viewpoint_node_2": 105.98, "workshop_right_entrance": 29.81},
    "to_viewpoint_node_2": {"to_viewpoint_node_1": 105.98, "to_viewpoint_node_3": 81.91, "teacher_dormitory_C": 104.23},
    "to_viewpoint_node_3": {"to_viewpoint_node_2": 81.91, "to_viewpoint_node_4": 132.33},
    "to_viewpoint_node_4": {"to_viewpoint_node_3": 132.33, "to_viewpoint_node_5": 139.35},
    "to_viewpoint_node_5": {"to_viewpoint_node_4": 139.35, "to_viewpoint_node_6": 81.23},
    "to_viewpoint_node_6": {"to_viewpoint_node_5": 81.23, "to_viewpoint_node_7": 103.88},
    "to_viewpoint_node_7": {"to_viewpoint_node_6": 103.88, "to_viewpoint_node_8": 83.01},
    "to_viewpoint_node_8": {"to_viewpoint_node_7": 83.01, "to_viewpoint_node_9": 131.54},
    "to_viewpoint_node_9": {"to_viewpoint_node_8": 131.54, "to_viewpoint_node_10": 115.65},
    "to_viewpoint_node_10": {"to_viewpoint_node_9": 115.65, "to_viewpoint_node_11": 121.32, "teacher_dormitory_D": 22.67},
    "to_viewpoint_node_11": {"to_viewpoint_node_10": 121.32, "to_viewpoint_node_12": 88.57},
    "to_viewpoint_node_12": {"to_viewpoint_node_11": 88.57, "to_viewpoint_node_13": 93.62},
    "to_viewpoint_node_13": {"to_viewpoint_node_12": 93.62, "view_point": 35.79},
    "to_dorms_junction_1": {"roundabout_left_junction": 298.24, "to_dorms_junction_2": 112.17},
    "to_dorms_junction_2": {"to_dorms_junction_1": 112.17, "to_dorms_junction_3": 72.76},
    "to_dorms_junction_3": {"to_dorms_junction_2": 72.76, "boys_dormitory_junction": 65.21},
    "boys_dormitory_junction": {"to_dorms_junction_3": 65.21, "to_boys_dormitory_A": 48.9, "to_boys_dormitory_B": 80.98, "before_cafeteria_junction": 331.63},
    "to_boys_dormitory_A": {"boys_dormitory_junction": 48.9, "boys_dormitory_A": 90.2},
    "to_boys_dormitory_B": {"boys_dormitory_junction": 80.98, "boys_dormitory_B": 70.23},
    "before_cafeteria_junction": {"boys_dormitory_junction": 331.63, "canteen": 82.17, "after_cafeteria_junction": 81.98, "stadium_entrance": 320.45},
    "after_cafeteria_junction": {"before_cafeteria_junction": 81.98, "canteen": 78.36, "girls_dormitory_junction": 232.89},
    "stadium_entrance": {"stadium": 41.66, "before_cafeteria_junction": 320.45},
    "girls_dormitory_junction": {"after_cafeteria_junction": 232.89, "girls_dormitory": 257.55},
}

PLACES = {
    "main entrance": {"node": "entrance", "type": "entrance", "aliases": ["entrance", "front gate", "gate a"]},
    "entrance b": {"node": "entrance_b", "type": "entrance", "aliases": ["back entrance", "gate b"]},
    "main building": {"node": "main_building", "type": "building", "aliases": ["academic building"]},
    "workshop": {"node": "workshop_left_entrance", "type": "building", "aliases": ["workshop building"]},
    "teacher dormitory a": {"node": "teacher_dormitory_A", "type": "dormitory", "aliases": ["teacher dorm a"]},
    "teacher dormitory b": {"node": "teacher_dormitory_B", "type": "dormitory", "aliases": ["teacher dorm b"]},
    "teacher dormitory c": {"node": "teacher_dormitory_C", "type": "dormitory", "aliases": ["teacher dorm c"]},
    "teacher dormitory d": {"node": "teacher_dormitory_D", "type": "dormitory", "aliases": ["teacher dorm d"]},
    "view point": {"node": "view_point", "type": "facility", "aliases": ["viewpoint", "view place"]},
    "boys dormitory a": {"node": "boys_dormitory_A", "type": "dormitory", "aliases": ["boys dorm a", "male dormitory a"]},
    "boys dormitory b": {"node": "boys_dormitory_B", "type": "dormitory", "aliases": ["boys dorm b", "male dormitory b"]},
    "canteen": {"node": "canteen", "type": "facility", "aliases": ["cafeteria", "food court", "dining hall", "restaurant"]},
    "girls dormitory": {"node": "girls_dormitory", "type": "dormitory", "aliases": ["girls dorm", "female dormitory"]},
    "stadium": {"node": "stadium", "type": "facility", "aliases": ["sports ground", "football field"]},
}

def dijkstra(start_node):
    distances = {node: float("inf") for node in GRAPH}
    previous = {node: None for node in GRAPH}
    distances[start_node] = 0
    queue = [(0, start_node)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in GRAPH[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    return distances, previous

def reconstruct_path(previous, destination_node):
    path = []
    current = destination_node

    while current is not None:
        path.append(current)
        current = previous[current]

    return list(reversed(path))

def determine_start_and_destination(sentence):
    pattern = r"(?:from\s+(.+?)\s+to\s+(.+?)|to\s+(.+?)\s+from\s+(.+?))(?:[?.!]|$)"

    match = re.search(pattern, sentence, re.IGNORECASE)
    if match:
        g1, g2, g3, g4 = match.groups()
         
        if g1 and g2:
            current_pos = g1
            destination = g2
        else:
            current_pos = g4
            destination = g3

        clean_filler = r"^(?:the|a|an)\s+"
        
        current_pos = re.sub(clean_filler, "", current_pos.strip(" .?!"), flags=re.IGNORECASE)
        destination = re.sub(clean_filler, "", destination.strip(" .?!"), flags=re.IGNORECASE)
                
        return current_pos, destination
    
    return None, None

def find_place(query):
    normalized = query.lower().strip()
    temp = []

    for place_name, place in PLACES.items():
        names = [place_name, *place["aliases"]]
        if any(name in normalized for name in names):
            temp.append((place_name, place))

    if len(temp) == 1:
        return temp[0]
    elif len(temp) == 2:
        _, destination = determine_start_and_destination(normalized)
        if temp[0][0] == destination:
            return temp[::-1]
        else:
            return temp
    elif len(temp) > 2:
        start_text, destination_text = determine_start_and_destination(normalized)
        if start_text and destination_text:
            start_match = next((item for item in temp if item[0] in start_text), None)
            destination_match = next((item for item in temp if item[0] in destination_text), None)
            if start_match and destination_match:
                return [start_match, destination_match]

    return None, None

def route_to_place(query, start_node="entrance"):
    found_places = find_place(query)
    place = None
    place_name = None

    if isinstance(found_places, tuple):
        place_name, place = found_places
    elif isinstance(found_places, list) and len(found_places) == 2:
        start_node = found_places[0][1]["node"]
        place_name, place = found_places[1]

    if place is None:
        return None

    destination_node = place["node"]
    distances, previous = dijkstra(start_node)
    path = reconstruct_path(previous, destination_node)

    if distances[destination_node] == float("inf"):
        return None

    distance = round(distances[destination_node], 2)
    walking_time_seconds = round(distance / WALKING_SPEED_FT_PER_SEC)

    return {
        "start": start_node,
        "startName": NODES[start_node]["label"],
        "destination": destination_node,
        "destinationName": NODES[destination_node]["label"],
        "matchedPlace": place_name,
        "distance": distance,
        "distanceUnit": "ft",
        "walkingTimeSeconds": walking_time_seconds,
        "walkingTimeText": format_walking_time(walking_time_seconds),
        "path": path,
        "points": [NODES[node] | {"id": node} for node in path],
    }

def format_walking_time(seconds):
    if seconds < 60:
        return f"{seconds} sec"

    minutes = seconds // 60
    remaining_seconds = seconds % 60

    if remaining_seconds == 0:
        return f"{minutes} min"

    return f"{minutes} min {remaining_seconds} sec"
