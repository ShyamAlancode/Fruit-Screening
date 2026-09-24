// ====================================================================
// CFRD: Optical Inspection Chamber Enclosure
// 3D CAD Script for OpenSCAD / FreeCAD
// License: Open Source (CFRD Project)
// Patent-Safe: Designed for 100% Offline Local Compilation
// ====================================================================

$fn = 60;

// Enclosure Dimensions (in mm)
box_w = 140;          // Width
box_d = 140;          // Depth
box_h = 180;          // Height
wall_t = 3.0;         // Wall Thickness

// Internal Optical Geometry
sensor_standoff = 22; // Height from apple apex to sensor lens
sensor_mount_w = 35;  // SparkFun Triad mount width
sensor_mount_d = 35;  // SparkFun Triad mount depth
cradle_radius = 45;   // Concave cradle for standard 65-85mm apples

module outer_shell() {
    difference() {
        // Main chamber outer body
        cube([box_w, box_d, box_h], center=true);
        
        // Hollow internal cavity
        translate([0, 0, wall_t])
            cube([box_w - 2*wall_t, box_d - 2*wall_t, box_h], center=true);
            
        // Front slide-in fruit access door opening
        translate([0, -box_d/2, -10])
            cube([100, 10, 130], center=true);
            
        // Top optical aperture for AS7265x sensor
        translate([0, 0, box_h/2 - wall_t])
            cylinder(h=wall_t*3, r=10, center=true);
            
        // Top camera aperture for Raspberry Pi Camera v2/v3
        translate([20, 0, box_h/2 - wall_t])
            cylinder(h=wall_t*3, r=6.5, center=true);
            
        // Ventilation and cable grommet (rear)
        translate([0, box_d/2, 60])
            cylinder(h=wall_t*3, r=8, center=true);
    }
}

module fruit_cradle() {
    // Concave ergonomic apple holder lined with neoprene foam
    translate([0, 0, -box_h/2 + wall_t + 12]) {
        difference() {
            cylinder(h=24, r=52, center=true);
            // Hemispherical depression
            translate([0, 0, 10])
                sphere(r=cradle_radius);
            // Center drainage / cleaning slot
            cylinder(h=30, r=6, center=true);
        }
    }
}

module sensor_mount_bracket() {
    // Mounting bracket for SparkFun AS7265x Triad board
    translate([0, 0, box_h/2 - wall_t - 6]) {
        difference() {
            cube([sensor_mount_w + 10, sensor_mount_d + 10, 6], center=true);
            
            // Optical central window (410-940nm sensor field-of-view)
            cylinder(h=10, r=8, center=true);
            
            // 4x M2.5 mounting standoffs matching SparkFun Triad PCB layout
            translate([ 11,  11, 0]) cylinder(h=10, r=1.35, center=true);
            translate([-11,  11, 0]) cylinder(h=10, r=1.35, center=true);
            translate([ 11, -11, 0]) cylinder(h=10, r=1.35, center=true);
            translate([-11, -11, 0]) cylinder(h=10, r=1.35, center=true);
        }
    }
}

module teflon_calibration_tray() {
    // Slide-in slot for 50x50mm PTFE Teflon 99% diffuse white tile
    translate([0, 0, box_h/2 - wall_t - 25]) {
        difference() {
            cube([65, 65, 4], center=true);
            cube([51, 51, 5], center=true); // 50x50mm tile pocket
        }
    }
}

// --------------------------------------------------------------------
// Assembly Rendering
// --------------------------------------------------------------------
color([0.25, 0.25, 0.25, 0.85]) outer_shell();
color([0.85, 0.25, 0.25, 1.00]) fruit_cradle();
color([0.20, 0.50, 0.90, 1.00]) sensor_mount_bracket();
color([0.95, 0.95, 0.95, 0.70]) teflon_calibration_tray();
