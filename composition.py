import matplotlib.pyplot as plt
import numpy as np

# Experimental data
WireStr = ['Al', 'Si', 'Cr', 'Fe', 'Ni', 'Zn', 'Mn', 'Others']
WireWtPercent = [13.18, 0.58, 18.33, 67.04, 0.08, 0.79, 0.00, 0.00]

GainStr = ['Si', 'Cr', 'Fe', 'Ni', 'Mn','others']
GainWtPercent = [0.42, 18.97, 72.79, 7.824, 0.0,0.0]

Kanthal = ['Al', 'Si', 'Cr', 'Fe', 'Ni', 'Zn', 'Mn', 'Others']
KanthalWtPercent = [6.00, 0.50, 22.00, 70.00, 0.00, 0.00, 0.50, 1.00]

StainlessSteel = ['Si', 'Cr', 'Fe', 'Ni', 'Mn','others']
StainlessSteelWtPercent = [0.50, 18.00, 70.00, 9.00, 1.50,1.00]

# Create the figure and subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2 rows, 2 columns

# Set y-axis limits for all subplots
axes[0, 0].set_ylim(0, 80)  # Wire composition
axes[0, 1].set_ylim(0, 80)  # Gain composition
axes[1, 0].set_ylim(0, 80)  # Kanthal composition
axes[1, 1].set_ylim(0, 80)  # Stainless Steel composition

# First subplot: Wire composition
axes[0, 0].bar(WireStr, WireWtPercent, color='skyblue')
for i, percent in enumerate(WireWtPercent):
    axes[0, 0].text(i, percent + 1, f'{percent:.2f}%', ha='center', fontsize=10)
axes[0, 0].set_xlabel('Elements')
axes[0, 0].set_ylabel('Weight Percentage (%)')
axes[0, 0].set_title('Wire (exp)')

# Second subplot: Gain composition
axes[0, 1].bar(GainStr, GainWtPercent, color='lightgreen')
for i, percent in enumerate(GainWtPercent):
    axes[0, 1].text(i, percent + 1, f'{percent:.2f}%', ha='center', fontsize=10)
axes[0, 1].set_xlabel('Elements')
axes[0, 1].set_ylabel('Weight Percentage (%)')
axes[0, 1].set_title('Sheath (exp)')

# Third subplot: Kanthal composition
axes[1, 0].bar(Kanthal, KanthalWtPercent, color='salmon')
for i, percent in enumerate(KanthalWtPercent):
    axes[1, 0].text(i, percent + 1, f'{percent:.2f}%', ha='center', fontsize=10)
axes[1, 0].set_xlabel('Elements')
axes[1, 0].set_ylabel('Weight Percentage (%)')
axes[1, 0].set_title('Kanthal')

# Fourth subplot: Stainless Steel composition
axes[1, 1].bar(StainlessSteel, StainlessSteelWtPercent, color='gold')
for i, percent in enumerate(StainlessSteelWtPercent):
    axes[1, 1].text(i, percent + 1, f'{percent:.2f}%', ha='center', fontsize=10)
axes[1, 1].set_xlabel('Elements')
axes[1, 1].set_ylabel('Weight Percentage (%)')
axes[1, 1].set_title('Stainless Steel')

# Adjust layout and show the plot
fig.suptitle('SEM-EDS Results of Wire and Sheath Composition', fontsize=16, y=1.02)
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()