EESchema Schematic File Version 2
LIBS:power
LIBS:device
LIBS:switches
LIBS:relays
LIBS:motors
LIBS:transistors
LIBS:conn
LIBS:linear
LIBS:regul
LIBS:74xx
LIBS:cmos4000
LIBS:adc-dac
LIBS:memory
LIBS:xilinx
LIBS:microcontrollers
LIBS:dsp
LIBS:microchip
LIBS:analog_switches
LIBS:motorola
LIBS:texas
LIBS:intel
LIBS:audio
LIBS:interface
LIBS:digital-audio
LIBS:philips
LIBS:display
LIBS:cypress
LIBS:siliconi
LIBS:opto
LIBS:atmel
LIBS:contrib
LIBS:valves
LIBS:691312510005
EELAYER 25 0
EELAYER END
$Descr A4 11693 8268
encoding utf-8
Sheet 1 1
Title ""
Date ""
Rev ""
Comp ""
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L Conn_01x05_Male J1
U 1 1 63B001C9
P 5200 3575
F 0 "J1" H 5315 3250 79  0000 C CNB
F 1 "sensor_foot" H 5305 3885 79  0000 C CNB
F 2 "Pin_Headers:Pin_Header_Straight_1x05_Pitch2.54mm" H 5200 3575 50  0001 C CNN
F 3 "" H 5200 3575 50  0001 C CNN
	1    5200 3575
	1    0    0    1   
$EndComp
$Comp
L 691312510005 J2
U 1 1 63B003CC
P 6305 3575
F 0 "J2" H 6145 3865 79  0000 L BNB
F 1 "extension_foot" H 5755 3245 79  0000 L BNB
F 2 "libs:691312510005" H 6305 3575 50  0001 L BNN
F 3 "" H 6305 3575 50  0001 L BNN
	1    6305 3575
	1    0    0    -1  
$EndComp
Wire Wire Line
	5400 3375 6105 3375
Wire Wire Line
	5400 3475 6105 3475
Wire Wire Line
	5400 3575 6105 3575
Wire Wire Line
	5400 3675 6105 3675
Wire Wire Line
	5400 3775 6105 3775
$EndSCHEMATC
