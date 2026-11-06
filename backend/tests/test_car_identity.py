"""Car identity (car_identity.py): model / class / game version per file."""

from app.car_identity import identify_car


def test_identify_from_vehicle_model():
    ident = identify_car({"VehicleModel": "BMW M4 LMGT3"}, "Team WRT 2026 #69:WEC")
    assert ident.model == "BMW M4 LMGT3"
    assert ident.car_class == "LMGT3"
    assert ident.game_version == "unknown"


def test_identify_from_livery_table():
    ident = identify_car({}, "Aston Martin THOR Team 2026 #007:WEC")
    assert ident.model == "Aston Martin Valkyrie"
    assert ident.car_class == "Hypercar"


def test_number_match_exact():
    ident = identify_car({}, "Toyota Racing 2026 #7:WEC")
    assert ident.model == "Toyota TR010"
    assert ident.car_class == "Hypercar"


def test_unknown_car():
    ident = identify_car({}, "Some Team 2026 #99:WEC")
    assert ident.model == "Some Team 2026 #99:WEC"
    assert ident.car_class == "unknown"


def test_vehicle_class_meta_wins_and_normalises():
    ident = identify_car({"VehicleModel": "Ferrari 499P", "VehicleClass": "HYPERCAR",
                          "GameVersion": "1.2.3"}, "x")
    assert ident.car_class == "Hypercar"
    assert ident.game_version == "1.2.3"


def test_vehicle_class_hyper_is_hypercar():
    # real recordings (Ferrari at COTA, BMW at Daytona) say "Hyper"
    for raw in ("Hyper", "HYPER", " hyper "):
        assert identify_car({"VehicleClass": raw}, "x").car_class == "Hypercar"
