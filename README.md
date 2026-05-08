so what I did was I calculated acuity for anacc classes G1-G4 and anacc classes1-13, and do the acuity_index pooled 0-1 and combine the  two dataset

for the model i did 4 versions balanced quality 40 percent experience 40 percent staffing 20 percent
Quality focused 60Q 30E 10S
Experience 30Q 60E 10S
Staffing 40Q 30E 30S

DEA model type output-oriented DEA model

Input Total Staff+acuity(1st model) only care staff+acuity(2nd model)

I used linear Programming

VRS, CRS and NIRS

Together with Pooled DEA O1 O2 and Seperate DEA O1 and O2 results we can do the meta-frontier analysis

What I calculated are Efficiency scores, rankings, scale efficiency, scale direction, benchmarking peers and improvement targets.

DEA Results Column Descriptions

facility_id
Unique identifier for each aged care facility.

group
Dataset grouping of the facility (e.g., O1 or O2).

model
Output weighting model used (e.g., Balanced, Quality-focused, etc.).

rts
Returns to scale assumption applied (e.g., VRS or CRS).

Efficiency Measures

efficiency
DEA efficiency score ranging from 0 to 1.
A value of 1 indicates that the facility is fully efficient and lies on the frontier, while values below 1 indicate inefficiency.

Output Performance

current_output
The facility’s current composite performance score based on weighted outputs.

target_output
The output level required for the facility to become efficient, given its current inputs.

output_gap
The absolute increase in output required to reach the efficiency frontier:
output_gap=target_output−current_output

output_gap_pct
The percentage increase in output required to reach the frontier.

Input Adjustment (Care Staff)

current_care_staff
Current number of care staff used by the facility.

target_care_staff
Optimal level of care staff based on efficient peer combinations.

care_staff_reduction
The amount of excess care staff that could be reduced without reducing output.

care_staff_reduction_pct
The percentage reduction in care staff required to operate efficiently.

Input Adjustment (Acuity Index)

current_acuity_index
Current level of resident acuity (care complexity), scaled between 0 and 1.

target_acuity_index
Benchmark acuity level implied by efficient peers.

acuity_index_reduction
The difference between current and benchmark acuity levels.

acuity_index_reduction_pct
The percentage difference in acuity relative to the benchmark.


