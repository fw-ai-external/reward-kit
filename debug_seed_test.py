#!/usr/bin/env python3
"""
Debug test to verify that generate_random_map produces different outputs with different seeds.
This will help us understand if the seed issue is in the random map generation or elsewhere.
"""

from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def test_seed_diversity():
    """Test if different seeds produce different maps."""
    print("🧪 Testing seed diversity with generate_random_map...")

    seeds = [42, 123, 456]
    maps = {}

    for seed in seeds:
        print(f"\n🌱 Testing seed {seed}:")
        desc = generate_random_map(size=4, p=0.8, seed=seed)
        map_str = "\n".join(["".join(row) for row in desc])
        maps[seed] = map_str
        print(f"Generated map:\n{map_str}")

    # Check if maps are different
    unique_maps = set(maps.values())
    print(f"\n📊 Results:")
    print(f"  • Seeds tested: {seeds}")
    print(f"  • Maps generated: {len(maps)}")
    print(f"  • Unique maps: {len(unique_maps)}")

    if len(unique_maps) == len(seeds):
        print("✅ SUCCESS: All seeds generated different maps!")
        return True
    else:
        print("❌ FAILURE: Some seeds generated identical maps!")
        print("Map details:")
        for seed, map_str in maps.items():
            print(f"  Seed {seed}: {repr(map_str)}")
        return False


def test_consecutive_calls():
    """Test if consecutive calls with the same seed produce the same map."""
    print("\n🔄 Testing consecutive calls with same seed...")

    seed = 42
    map1 = generate_random_map(size=4, p=0.8, seed=seed)
    map2 = generate_random_map(size=4, p=0.8, seed=seed)

    map1_str = "\n".join(["".join(row) for row in map1])
    map2_str = "\n".join(["".join(row) for row in map2])

    print(f"First call with seed {seed}:\n{map1_str}")
    print(f"Second call with seed {seed}:\n{map2_str}")

    if map1_str == map2_str:
        print("✅ SUCCESS: Same seed produces same map!")
        return True
    else:
        print("❌ FAILURE: Same seed produces different maps!")
        return False


if __name__ == "__main__":
    print("🔍 Debugging seed behavior for FrozenLake map generation")
    print("=" * 60)

    # Test 1: Different seeds should produce different maps
    diversity_ok = test_seed_diversity()

    # Test 2: Same seed should produce same map
    consistency_ok = test_consecutive_calls()

    print("\n" + "=" * 60)
    print("🏁 Final Results:")
    print(f"  • Seed diversity: {'✅ PASS' if diversity_ok else '❌ FAIL'}")
    print(f"  • Seed consistency: {'✅ PASS' if consistency_ok else '❌ FAIL'}")

    if diversity_ok and consistency_ok:
        print("🎉 generate_random_map is working correctly!")
        print("🔍 The issue is likely in seed propagation, not map generation.")
    else:
        print("⚠️  generate_random_map has issues - this may be the root cause.")
