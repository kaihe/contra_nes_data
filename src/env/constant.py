"""Contra NES constants: RAM addresses and static lookup tables.

Sourced from Contra-Nes/data.json and docs/Enemy Glossary.md.
"""

# ── RAM addresses ────────────────────────────────────────────────────────────
ADDR_LIVES          = 50     # |i1   player lives
ADDR_LEVEL          = 48     # |u1   current level (0-indexed)
ADDR_WEAPON         = 0xAA   # |u1   bit4 = rapid fire flag; low nibble = weapon type
ADDR_INVINCIBILITY  = 0xB0   # |u1   INVINCIBILITY_TIMER — Barrier (B) weapon; set to
                             #       0x80 on pickup (0x90 on L7), decrements every 8 frames
ADDR_FALCON         = 0x7D   # |u1   FALCON_FLASH_TIMER — Falcon (Mega Shell) item; set to
                             #       0x20 on pickup, decrements per frame (screen-flash)
ADDR_GUN_ANI        = 0x010A # |u1   0x1F on the frame a gun powerup is collected
ADDR_XSCROLL        = 100    # >u2   horizontal scroll position (big-endian, 2 bytes)
ADDR_ENEMY_TYPE     = 0x528  # |u1   start of 16-slot enemy type array  ($0528)
ADDR_ENEMY_HP       = 1400   # |u1   start of 16-slot enemy HP array    ($0578)
ADDR_ENEMY_HP_COUNT = 16     #       number of enemy HP slots
ADDR_ENEMY_ATTR     = 0x5A8  # |u1   start of 16-slot ENEMY_ATTRIBUTES array ($05a8);
                             #       for a weapon item (type 0x00) low 3 bits = which weapon
ADDR_INDOOR_CLEARED = 0x37   # |u1   INDOOR_SCREEN_CLEARED
ADDR_XSCROLL_HI     = 0x64   # |u1   LEVEL_SCREEN_NUMBER (indoor/vertical screen index)
ADDR_SCROLL_OFF     = 0x65   # |u1   LEVEL_SCREEN_SCROLL_OFFSET: pixels within screen (wraps at 0xF0)
ADDR_PLAYER_ADV     = 0xd0   # |u1   INDOOR_PLAYER_ADV_FLAG: set while walking through a door
ADDR_PLAYER_STATE   = 0x90   # |u1   PLAYER_STATE — 0x02 == dead (any cause, incl. pit falls)
ADDR_PLAYER_DEATH   = 180    # 0xB4 bit0 == PLAYER_DEATH_FLAG (enemy hit)
ADDR_LEVEL_STOP_SCROLL = 0x58  # |u1   level length while playing; 0xff once boss auto-scroll starts
ADDR_LEVEL_ROUTINE  = 0x2c   # |u1   LEVEL_ROUTINE_INDEX (0x04 == active gameplay)

ENEMY_TYPE_SOLDIER      = 0x05  # 1-hp swarming infantry (the common outdoor grunt)
ENEMY_TYPE_FALLING_ROCK = 0x13  # respawns endlessly from the rock cave on L3

# ── Lookup tables ────────────────────────────────────────────────────────────
WEAPON_NAMES = {0: "Regular", 1: "MachineGun", 2: "Flamethrower",
                3: "Spread",  4: "Laser"}

# Weapon-item flavour, from a dropped item's `ENEMY_ATTRIBUTES & 0x07`. The four
# gun items (M/F/S/L) carry the same code as the player weapon nibble they grant
# (1/2/3/4), so a gun item's attr equals the gun it turns into; R/B/Falcon are
# augments that don't change the gun nibble.
ITEM_WEAPON_NAMES = {0: "Rapid", 1: "MachineGun", 2: "Flamethrower",
                     3: "Spread", 4: "Laser", 5: "Barrier", 6: "Falcon"}
ITEM_GUN_ATTRS = {1, 2, 3, 4}   # attrs that grant a gun (change the weapon nibble)

# Common enemy types 0x00-0x0F (shared across all levels).
ENEMY_TYPE_NAMES_COMMON = {
    0x00: "Weapon Item",     0x01: "Bullet",         0x02: "Pill Box Sensor",
    0x03: "Flying Capsule",  0x04: "Rotating Gun",   0x05: "Soldier",
    0x06: "Sniper",          0x07: "Red Turret",     0x08: "Wall Cannon",
    0x09: "Unused",          0x0A: "Wall Plating",   0x0B: "Mortar Shot",
    0x0C: "Scuba Diver",     0x0D: "Unused",         0x0E: "Basquez",
    0x0F: "Basquez Bullet",
}

# Level-specific enemy types 0x10+, keyed by 0-indexed level. The same type byte
# names a different enemy on each level, so these must always be read with a level.
ENEMY_TYPE_NAMES_BY_LEVEL = {
    0: {  # L1 — Jungle
        0x10: "Bomb Turret", 0x11: "Plated Door", 0x12: "Exploding Bridge",
    },
    1: {  # L2 — Indoor Base
        0x10: "Boss Eye", 0x11: "Roller", 0x12: "Grenade", 0x13: "Wall Turret",
        0x14: "Core", 0x15: "Indoor Soldier", 0x16: "Jumping Soldier",
        0x17: "Grenade Launcher", 0x18: "Group of Four Soldiers",
        0x19: "Indoor Soldier Generator", 0x1A: "Indoor Roller Generator",
        0x1B: "Boss Eye Fire Ring Projectile", 0x1C: "Godomuga",
        0x1D: "Gardegura", 0x1E: "Garth", 0x1F: "Rangel",
        0x20: "Garth and Rangel Generator",
    },
    2: {  # L3 — Waterfall
        0x10: "Floating Rock Platform", 0x11: "Moving Flame", 0x12: "Rock Cave",
        0x13: "Falling Rock", 0x14: "Dragon", 0x15: "Dragon Tentacle Orb",
    },
    3: {  # L4 — Indoor Base (same enemy set as L2)
        0x10: "Boss Eye", 0x11: "Roller", 0x12: "Grenade", 0x13: "Wall Turret",
        0x14: "Core", 0x15: "Indoor Soldier", 0x16: "Jumping Soldier",
        0x17: "Grenade Launcher", 0x18: "Group of Four Soldiers",
        0x19: "Indoor Soldier Generator", 0x1A: "Indoor Roller Generator",
        0x1B: "Boss Eye Fire Ring Projectile", 0x1C: "Godomuga",
        0x1D: "Gardegura", 0x1E: "Garth", 0x1F: "Rangel",
        0x20: "Garth and Rangel Generator",
    },
    4: {  # L5 — Snow Field
        0x10: "Ice Grenade Generator", 0x11: "Ice Grenade", 0x12: "Tank",
        0x13: "Ice Separator", 0x14: "Alien Carrier", 0x15: "Flying Saucer",
        0x16: "Drop Bomb",
    },
    5: {  # L6 — Energy Zone
        0x10: "Fire Beam", 0x11: "Fire Beam", 0x12: "Fire Beam",
        0x13: "Giant Boss Soldier", 0x14: "Spiked Projectile",
    },
    6: {  # L7 — Hangar
        0x10: "Claw", 0x11: "Rising Spiked Wall", 0x12: "Tall Spiked Wall",
        0x13: "Mining Cart Generator", 0x14: "Mining Cart",
        0x15: "Stationary Mining Cart", 0x16: "Armored Door",
        0x17: "Mortar Launcher", 0x18: "Soldier Generator",
    },
    7: {  # L8 — Alien's Lair
        0x10: "Emperor Demon Dragon God Java", 0x11: "Bundle",
        0x12: "Wadder", 0x13: "Poisonous Insect Gel",
        0x14: "Bugger", 0x15: "Eggron", 0x16: "Gomeramos King",
    },
}

# HP-bearing bosses / finite boss-objective components, per 0-indexed level.
BOSS_ENEMY_TYPES_COMMON = {0x0A}          # Wall Plating: shielded indoor cores
BOSS_ENEMY_TYPES_BY_LEVEL = {
    0: {0x10, 0x11}, 1: {0x10, 0x14}, 2: {0x14, 0x15}, 3: {0x10, 0x14, 0x1C},
    4: {0x12, 0x14}, 5: {0x13}, 6: {0x16, 0x17}, 7: {0x10, 0x15, 0x16},
}

# 0-indexed levels whose boss room does not auto-scroll into view (indoor bases).
INSIDE_LEVELS = {1, 3}


def enemy_name(etype: int, level: int | None = None) -> str:
    """Normalised task-label name for an enemy type, e.g. 0x03 -> 'flying_capsule'.

    Types 0x00-0x0F are common; 0x10+ are level-specific, so ``level`` (0-indexed)
    is needed to name them. Unknown types fall back to a ``0x??`` hex string.
    """
    table = (ENEMY_TYPE_NAMES_COMMON if etype <= 0x0F
             else ENEMY_TYPE_NAMES_BY_LEVEL.get(level or 0, {}))
    return table.get(etype, f"0x{etype:02x}").lower().replace(" ", "_")
