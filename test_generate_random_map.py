#!/usr/bin/env python3
"""
Test generate_random_map with different seeds

This script tests whether gymnasium's generate_random_map function
actually creates different maps when given different seeds.
"""

from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def test_random_map_generation():
    """Test that different seeds generate different maps."""
    print("🧪 Testing generate_random_map with different seeds...")
    print()

    # Test seeds that our diagnostic showed were identical
    test_seeds = [42, 123, 456]

    maps = {}

    for seed in test_seeds:
        print(f"🌱 Testing seed {seed}:")

        # Generate map with this seed
        desc = generate_random_map(size=4, p=0.8, seed=seed)

        # Convert to string for comparison
        map_str = "\n".join(["".join(row) for row in desc])

        maps[seed] = {"desc": desc, "map_str": map_str, "hash": hash(map_str)}

        print(f"   Map:\n{map_str}")
        print(f"   Hash: {hash(map_str)}")
        print()

    # Check for uniqueness
    map_strings = [info["map_str"] for info in maps.values()]
    unique_maps = len(set(map_strings))

    print("📊 Results:")
    print(f"   Total seeds tested: {len(test_seeds)}")
    print(f"   Unique maps generated: {unique_maps}")
    print(f"   Expected unique maps: {len(test_seeds)}")

    if unique_maps == len(test_seeds):
        print("   ✅ SUCCESS: All seeds generated different maps!")
        return True
    else:
        print("   ❌ ISSUE: Some seeds generated identical maps!")

        # Show which seeds generated the same maps
        seen_maps = {}
        for seed, info in maps.items():
            map_str = info["map_str"]
            if map_str in seen_maps:
                print(
                    f"   🔍 Seeds {seen_maps[map_str]} and {seed} generated identical maps"
                )
            else:
                seen_maps[map_str] = seed

        return False


def test_numpy_random_state():
    """Test if numpy random state affects map generation."""
    print("\n🔬 Testing numpy random state impact...")

    import numpy as np

    # Reset numpy random state and test
    np.random.seed(12345)  # Set a different numpy seed
    map1 = generate_random_map(size=4, p=0.8, seed=42)

    np.random.seed(67890)  # Set another different numpy seed
    map2 = generate_random_map(size=4, p=0.8, seed=42)

    map1_str = "\n".join(["".join(row) for row in map1])
    map2_str = "\n".join(["".join(row) for row in map2])

    print(f"   Same seed (42) with different numpy states:")
    print(f"   Map 1 (numpy seed 12345):\n{map1_str}")
    print(f"   Map 2 (numpy seed 67890):\n{map2_str}")

    if map1_str == map2_str:
        print(
            "   ✅ Maps are identical - generate_random_map properly isolates randomness"
        )
        return True
    else:
        print("   ❌ Maps are different - numpy global state affects generation!")
        return False


def test_consecutive_calls():
    """Test consecutive calls with same seed."""
    print("\n🔁 Testing consecutive calls with same seed...")

    maps = []
    for i in range(3):
        desc = generate_random_map(size=4, p=0.8, seed=42)
        map_str = "\n".join(["".join(row) for row in desc])
        maps.append(map_str)
        print(f"   Call {i+1} (seed 42):\n{map_str}")
        print()

    # Check if all calls produce the same result
    unique_consecutive = len(set(maps))

    if unique_consecutive == 1:
        print("   ✅ All consecutive calls with same seed produce identical maps")
        return True
    else:
        print(f"   ❌ Consecutive calls produced {unique_consecutive} different maps!")
        return False


if __name__ == "__main__":
    print("🔍 === GENERATE_RANDOM_MAP TESTING ===")
    print()

    # Run tests
    test1_passed = test_random_map_generation()
    test2_passed = test_numpy_random_state()
    test3_passed = test_consecutive_calls()

    print("\n📋 Summary:")
    print(f"   Different seeds test: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Numpy isolation test: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"   Consecutive calls test: {'✅ PASS' if test3_passed else '❌ FAIL'}")

    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 All tests passed! generate_random_map works correctly.")
    else:
        print(
            "\n⚠️ Some tests failed. There may be an issue with generate_random_map or our usage."
        )
