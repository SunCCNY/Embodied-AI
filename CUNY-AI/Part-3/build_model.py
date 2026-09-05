#!/usr/bin/env python3
"""Build a fixed-pelvis Unitree R1 with genuine left and right Revo 2 hands."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
R1_SOURCE = ROOT / "model/r1/r1_source.xml"
RIGHT_SOURCE = ROOT / "model/revo2/right/brainco-righthand-v2.xml"
LEFT_SOURCE = ROOT / "model/revo2/left/brainco-lefthand-v2.xml"
OUTPUT = ROOT / "model/r1_revo2_bimanual.xml"

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def rewrite_mesh_paths(asset: ET.Element, prefix: str) -> None:
    for mesh in asset.findall("mesh"):
        file_name = mesh.get("file")
        if file_name:
            mesh.set("file", f"{prefix}/{Path(file_name).name}")


def joint_gains(name: str) -> tuple[float, float, float]:
    if "ankle" in name:
        return 100.0, 8.0, 80.0
    if any(token in name for token in ("hip", "knee", "waist")):
        return 220.0, 14.0, 120.0
    if any(token in name for token in ("shoulder_pitch", "shoulder_roll")):
        return 160.0, 12.0, 120.0
    return 120.0, 10.0, 90.0


def is_hand_joint(name: str) -> bool:
    return any(name.startswith(f"{side}_{finger}") for side in ("left", "right") for finger in FINGERS)


def add_r1_actuators(root: ET.Element, actuator: ET.Element) -> None:
    for joint in root.findall(".//worldbody//joint"):
        name = joint.get("name", "")
        if not name or is_hand_joint(name) or joint.get("type") == "free":
            continue
        joint_range = joint.get("range")
        if not joint_range:
            continue
        kp, kv, force = joint_gains(name)
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"r1_{name}",
                "joint": name,
                "kp": f"{kp:g}",
                "kv": f"{kv:g}",
                "ctrlrange": joint_range,
                "forcerange": f"-{force:g} {force:g}",
            },
        )


def append_unique_assets(destination: ET.Element, source: ET.Element, prefix: str) -> None:
    copied = deepcopy(source)
    rewrite_mesh_paths(copied, prefix)
    existing = {
        (child.tag, child.get("name"))
        for child in destination
        if child.get("name")
    }
    for child in list(copied):
        key = (child.tag, child.get("name"))
        if child.get("name") and key in existing:
            continue
        destination.append(child)
        if child.get("name"):
            existing.add(key)


def remove_stock_hand(wrist: ET.Element, asset: ET.Element, side: str) -> None:
    wrist_mesh = f"{side}_wrist_roll_link"
    collision = f"{side}_hand_collision"
    for child in list(wrist):
        if child.tag == "geom" and (
            child.get("mesh") == wrist_mesh or child.get("name") == collision
        ):
            wrist.remove(child)
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") == wrist_mesh:
            asset.remove(mesh)


def add_adapter(wrist: ET.Element, side: str) -> ET.Element:
    ET.SubElement(
        wrist,
        "geom",
        {
            "name": f"{side}_revo2_wrist_sleeve",
            "type": "cylinder",
            "fromto": "0 0 0 0.105 0 0",
            "size": "0.035",
            "rgba": "0.79216 0.81961 0.93333 1",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        },
    )
    ET.SubElement(
        wrist,
        "geom",
        {
            "name": f"{side}_revo2_flange_adapter",
            "type": "cylinder",
            "fromto": "0.105 0 0 0.132 0 0",
            "size": "0.028",
            "rgba": "0.32 0.36 0.44 1",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        },
    )
    return ET.SubElement(
        wrist,
        "body",
        {
            "name": f"{side}_revo2_mount",
            "pos": "0.132 0 0",
            "quat": "0.70710678 0 0.70710678 0",
        },
    )


def mount_hand(
    worldbody: ET.Element,
    asset: ET.Element,
    hand_root: ET.Element,
    side: str,
) -> None:
    wrist = worldbody.find(f".//body[@name='{side}_wrist_roll_link']")
    hand_worldbody = hand_root.find("worldbody")
    hand_base = None if hand_worldbody is None else hand_worldbody.find(f"body[@name='{side}_base_link']")
    if wrist is None or hand_base is None:
        raise RuntimeError(f"Could not resolve the {side} wrist or Revo 2 base")

    remove_stock_hand(wrist, asset, side)
    mount = add_adapter(wrist, side)
    hand_copy = deepcopy(hand_base)
    hand_copy.set("pos", "0 0 0")
    hand_copy.set("quat", "1 0 0 0")

    existing_site = hand_copy.find("site[@name='grasp_site']")
    if existing_site is not None:
        hand_copy.remove(existing_site)
    sign = 1.0 if side == "right" else -1.0
    ET.SubElement(
        hand_copy,
        "site",
        {
            "name": f"{side}_grasp_site",
            "pos": f"0.030 {sign * 0.0092:.4f} 0.062",
            "size": "0.008",
            "rgba": "0.1 1 0.2 0.35",
            "group": "4",
        },
    )
    ET.SubElement(
        hand_copy,
        "site",
        {
            "name": f"{side}_acquire_site",
            "pos": f"0.030 {sign * 0.020:.4f} 0.085",
            "size": "0.006",
            "rgba": "0.2 0.65 1 0.30",
            "group": "4",
        },
    )
    mount.append(hand_copy)


def add_scene(worldbody: ET.Element, equality: ET.Element) -> None:
    ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "1.2 -1.0 2.1", "dir": "-0.6 0.4 -1", "directional": "true", "castshadow": "true"})
    ET.SubElement(worldbody, "light", {"name": "fill_light", "pos": "-0.6 0.4 1.5", "dir": "0.4 -0.2 -1", "directional": "true", "diffuse": "0.35 0.42 0.55"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "size": "2 2 0.05", "rgba": "0.12 0.14 0.18 1", "friction": "0.9 0.01 0.001"})

    anchor = ET.SubElement(worldbody, "body", {"name": "ball_anchor", "mocap": "true", "pos": "0.42 -0.28 0.88"})
    ET.SubElement(anchor, "geom", {"name": "ball_anchor_marker", "type": "sphere", "size": "0.008", "rgba": "0.1 0.9 0.25 0.25", "contype": "0", "conaffinity": "0"})
    ball = ET.SubElement(worldbody, "body", {"name": "ball", "pos": "0.42 -0.28 0.88"})
    ET.SubElement(ball, "freejoint", {"name": "ball_freejoint"})
    ET.SubElement(ball, "geom", {"name": "ball_geom", "type": "sphere", "size": "0.018", "mass": "0.05", "rgba": "0.96 0.28 0.08 1", "friction": "1.1 0.01 0.001", "condim": "4", "priority": "2"})
    ET.SubElement(ball, "site", {"name": "ball_site", "size": "0.004", "rgba": "1 0.6 0 1"})

    handoff = ET.SubElement(worldbody, "body", {"name": "handoff_marker", "mocap": "true", "pos": "0.39 0 0.98"})
    ET.SubElement(handoff, "geom", {"name": "handoff_target_marker", "type": "sphere", "size": "0.028", "rgba": "0.15 0.9 0.35 0.16", "contype": "0", "conaffinity": "0"})
    target = ET.SubElement(worldbody, "body", {"name": "throw_target", "mocap": "true", "pos": "0.82 0.22 0.78"})
    ET.SubElement(target, "geom", {"name": "throw_target_marker", "type": "cylinder", "size": "0.10 0.008", "quat": "0.707107 0 0.707107 0", "rgba": "0.12 0.55 1 0.28", "contype": "0", "conaffinity": "0"})
    ET.SubElement(equality, "weld", {"name": "ball_fixture", "body1": "ball", "body2": "ball_anchor", "solref": "0.012 1", "active": "true"})


def build() -> None:
    r1_tree = ET.parse(R1_SOURCE)
    right_tree = ET.parse(RIGHT_SOURCE)
    left_tree = ET.parse(LEFT_SOURCE)
    root = r1_tree.getroot()
    right_root = right_tree.getroot()
    left_root = left_tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", ".")
    compiler.set("autolimits", "true")
    compiler.set("discardvisual", "false")
    option = root.find("option")
    if option is None:
        option = ET.Element("option", {"timestep": "0.0025", "integrator": "implicitfast", "iterations": "32", "ls_iterations": "8", "gravity": "0 0 -9.81"})
        root.insert(1, option)
    visual = ET.Element("visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.75 0.75 0.75", "ambient": "0.22 0.22 0.22", "specular": "0.4 0.4 0.4"})
    ET.SubElement(visual, "global", {"azimuth": "145", "elevation": "-16", "offwidth": "960", "offheight": "720"})
    root.insert(2, visual)

    for hand_root in (right_root, left_root):
        for default in hand_root.findall("default"):
            copied_default = deepcopy(default)
            if hand_root is left_root:
                for nested in copied_default.findall(".//default[@class='visual_hand']"):
                    nested.set("class", "visual_lefthand")
            root.append(copied_default)

    asset = root.find("asset")
    right_asset = right_root.find("asset")
    left_asset = left_root.find("asset")
    if asset is None or right_asset is None or left_asset is None:
        raise RuntimeError("Every source model must contain an asset section")
    rewrite_mesh_paths(asset, "r1/assets")
    append_unique_assets(asset, right_asset, "revo2/right/meshes")
    append_unique_assets(asset, left_asset, "revo2/left/meshes")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("R1 source is missing worldbody")
    floating = worldbody.find(".//joint[@name='floating_base_joint']")
    if floating is not None:
        for parent in worldbody.iter():
            if floating in list(parent):
                parent.remove(floating)
                break
    mount_hand(worldbody, asset, right_root, "right")
    mount_hand(worldbody, asset, left_root, "left")

    valid_bodies = {body.get("name", "") for body in root.findall(".//body") if body.get("name")}
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for exclude in list(contact.findall("exclude")):
        if exclude.get("body1") not in valid_bodies or exclude.get("body2") not in valid_bodies:
            contact.remove(exclude)
    for hand_root in (right_root, left_root):
        hand_contact = hand_root.find("contact")
        if hand_contact is not None:
            for child in list(hand_contact):
                contact.append(deepcopy(child))

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    add_r1_actuators(root, actuator)
    for hand_root in (right_root, left_root):
        hand_actuator = hand_root.find("actuator")
        if hand_actuator is not None:
            for child in list(hand_actuator):
                actuator.append(deepcopy(child))

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for hand_root in (right_root, left_root):
        hand_equality = hand_root.find("equality")
        if hand_equality is not None:
            for child in list(hand_equality):
                equality.append(deepcopy(child))
    add_scene(worldbody, equality)

    root.set("model", "Unitree R1 dual BrainCo Revo 2 handoff throw")
    ET.indent(r1_tree, space="  ")
    r1_tree.write(OUTPUT, encoding="utf-8", xml_declaration=False)
    print(OUTPUT)


if __name__ == "__main__":
    build()
