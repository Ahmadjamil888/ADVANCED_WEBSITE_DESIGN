# Visual Guide - Dashboard Design

## Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  SIDEBAR          HEADER                                        │
│  ┌────────┐  ┌──────────────────────────────────────────────┐  │
│  │ 🤖     │  │ Trained Models          [+ Create] [👤] [⚙️] │  │
│  │ 🧠     │  └──────────────────────────────────────────────┘  │
│  │ 📊     │                                                     │
│  │ ⚙️     │  CONTENT AREA                                       │
│  │ 💳  ✕  │  ┌──────────────────────────────────────────────┐  │
│  │        │  │                                              │  │
│  │        │  │  [Model Card]  [Model Card]  [Model Card]   │  │
│  │        │  │                                              │  │
│  │        │  │  [Model Card]  [Model Card]  [Model Card]   │  │
│  │        │  │                                              │  │
│  │        │  │  [Model Card]  [Model Card]  [Model Card]   │  │
│  │        │  │                                              │  │
│  │        │  └──────────────────────────────────────────────┘  │
│  │        │                                                     │
│  └────────┘                                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Color Reference

### Background Gradient
```
#0a0a0a (Deep Black)
    ↓
#0f0f1a (Dark Blue-Black)
```

### Purple Accent Gradient
```
#1a0033 (Deep Purple)
    ↓
#330066 (Medium Purple)
    ↓
#4d0099 (Bright Purple on Hover)
```

### Text Colors
```
#ffffff - Primary Text (White)
#cccccc - Secondary Text (Light Gray)
#999999 - Tertiary Text (Medium Gray)
```

---

## Component Visual Examples

### Button States

#### Default State
```
┌─────────────────────────┐
│  + Create AI Model      │  ← Purple Gradient
└─────────────────────────┘
```

#### Hover State
```
┌─────────────────────────┐
│  + Create AI Model      │  ← Brighter Gradient + Glow
└─────────────────────────┘
  ✨ Glow Effect
```

---

### Card States

#### Default State
```
┌──────────────────────────┐
│ Model Name               │
│ ─────────────────────── │
│ Status: Training         │
│ Created: 2 days ago      │
│                          │
│ [View] [Delete] [Edit]   │
└──────────────────────────┘
```

#### Hover State
```
┌──────────────────────────┐
│ Model Name               │  ↑ Lifted 4px
│ ─────────────────────── │  ✨ Glow Effect
│ Status: Training         │  ✨ Brighter Border
│ Created: 2 days ago      │
│                          │
│ [View] [Delete] [Edit]   │
└──────────────────────────┘
```

---

### Sidebar States

#### Collapsed State
```
┌──┐
│🤖│
│🧠│
│📊│
│⚙️│
│💳│
└──┘
```

#### Expanded State
```
┌──────────────┐
│🤖 Models     │
│🧠 LLMs       │
│📊 Datasets   │
│⚙️ In Progress│
│💳 Billing    │
│         [✕]  │
└──────────────┘
```

---

### Sidebar Item States

#### Default
```
┌─────────────────────┐
│ 🤖 Trained Models   │  ← Gray text
└─────────────────────┘
```

#### Hover
```
┌─────────────────────┐
│ 🤖 Trained Models   │  ← White text
└─────────────────────┘  ← Purple tint background
  ← Purple left border
```

#### Active
```
┌─────────────────────┐
│ 🤖 Trained Models   │  ← White text
└─────────────────────┘  ← Purple gradient background
  ← Bright purple left border
```

---

## Animation Examples

### Fade In Animation
```
Frame 1:  Model Card (opacity: 0, translateY: 10px)
Frame 2:  Model Card (opacity: 0.5, translateY: 5px)
Frame 3:  Model Card (opacity: 1, translateY: 0px)
```

### Hover Lift Effect
```
Before:   Card at Y: 0px
Hover:    Card at Y: -4px (lifted up)
Transition: 0.3s ease
```

### Glow Effect
```
Before:   box-shadow: none
Hover:    box-shadow: 0 0 20px rgba(147, 51, 234, 0.3)
Transition: 0.3s ease
```

---

## Typography Hierarchy

### Page Title
```
Trained Models
═══════════════════════════════════════════════════════════
1.8rem, Bold, White
```

### Section Header
```
Current Plan: PRO
─────────────────────────────────────────────────────────
1.5rem, Bold, White
```

### Card Title
```
Model Name
─────────────────────────────────────────────────────────
1.2rem, Semi-bold, White
```

### Body Text
```
Status: Training
Created: 2 days ago
─────────────────────────────────────────────────────────
1rem, Regular, Light Gray
```

### Small Text
```
Last updated: 2 hours ago
─────────────────────────────────────────────────────────
0.9rem, Regular, Medium Gray
```

---

## Spacing Examples

### Card Padding
```
┌─────────────────────────────┐
│  1.5rem                     │
│  ┌───────────────────────┐  │
│  │ Model Name            │  │
│  │ Status: Training      │  │
│  └───────────────────────┘  │
│  1.5rem                     │
└─────────────────────────────┘
```

### Grid Gap
```
┌──────────┐  1.5rem  ┌──────────┐  1.5rem  ┌──────────┐
│ Card 1   │          │ Card 2   │          │ Card 3   │
└──────────┘          └──────────┘          └──────────┘
```

### Sidebar Item Padding
```
┌─────────────────────┐
│ 0.75rem             │
│  🤖 Trained Models  │
│ 0.75rem             │
└─────────────────────┘
```

---

## Responsive Breakpoints

### Mobile (< 640px)
```
┌─────────────────────┐
│ [☰] Trained Models  │
├─────────────────────┤
│ [Model Card]        │
│ [Model Card]        │
│ [Model Card]        │
└─────────────────────┘
```

### Tablet (640px - 1024px)
```
┌──┬──────────────────────────┐
│  │ Trained Models           │
│  ├──────────────────────────┤
│  │ [Card] [Card]            │
│  │ [Card] [Card]            │
│  │ [Card] [Card]            │
└──┴──────────────────────────┘
```

### Desktop (> 1024px)
```
┌──┬────────────────────────────────┐
│  │ Trained Models                 │
│  ├────────────────────────────────┤
│  │ [Card] [Card] [Card]           │
│  │ [Card] [Card] [Card]           │
│  │ [Card] [Card] [Card]           │
└──┴────────────────────────────────┘
```

---

## Color Swatches

### Primary Colors
```
████████████ #0a0a0a - Deep Black (Background)
████████████ #0f0f1a - Dark Gradient (Background End)
████████████ #000000 - Pure Black (Text/Borders)
```

### Accent Colors
```
████████████ #9333ea - Deep Purple (Primary Accent)
████████████ #1a0033 - Purple Gradient Start
████████████ #330066 - Purple Gradient End
████████████ #4d0099 - Bright Purple (Hover)
```

### Neutral Colors
```
████████████ #ffffff - White (Primary Text)
████████████ #cccccc - Light Gray (Secondary Text)
████████████ #999999 - Medium Gray (Tertiary Text)
████████████ #333333 - Dark Gray (Borders)
```

### Semantic Colors
```
████████████ #00ff00 - Green (Success/Current)
████████████ #ff0000 - Red (Error/Delete)
████████████ #ffaa00 - Orange (Warning)
```

---

## Border Radius Reference

### Small Radius (4px)
```
┌─────────┐
│ Button  │  ← Slightly rounded
└─────────┘
```

### Medium Radius (6px)
```
┌─────────────┐
│ Button      │  ← More rounded
└─────────────┘
```

### Large Radius (8px)
```
┌─────────────────┐
│ Card            │  ← Significantly rounded
└─────────────────┘
```

---

## Shadow Effects

### Subtle Shadow
```
┌─────────────┐
│ Element     │  ← 0 1px 2px rgba(0,0,0,0.05)
└─────────────┘
```

### Medium Shadow
```
┌─────────────┐
│ Element     │  ← 0 4px 6px rgba(0,0,0,0.1)
└─────────────┘
```

### Glow Shadow
```
┌─────────────┐
│ Element     │  ← 0 0 20px rgba(147,51,234,0.3)
└─────────────┘
  ✨ Purple Glow
```

---

## Transition Timings

### Fast (0.2s)
```
Hover effect on sidebar items
Quick visual feedback
```

### Normal (0.3s)
```
Button hover effects
Card transitions
Sidebar expand/collapse
```

### Slow (0.5s)
```
Page load animations
Fade in effects
Large transitions
```

---

## Accessibility Features

### Focus State
```
┌─────────────────────┐
│ Button              │
└─────────────────────┘
  ↑ 2px outline
  ↑ 2px offset
```

### High Contrast
```
Text: #ffffff on #0a0a0a
Ratio: 21:1 (AAA compliant)

Interactive: #9333ea on #0a0a0a
Ratio: 5.5:1 (AA compliant)
```

### Keyboard Navigation
```
Tab → Focus next element
Shift+Tab → Focus previous element
Enter/Space → Activate button
Escape → Close modal/sidebar
```

---

## Animation Timing Functions

### Ease
```
Start: Slow
Middle: Fast
End: Slow
Best for: General UI animations
```

### Ease-In
```
Start: Slow
End: Fast
Best for: Entering animations
```

### Ease-Out
```
Start: Fast
End: Slow
Best for: Exiting animations
```

### Linear
```
Constant speed throughout
Best for: Continuous animations
```

---

## Best Practices

✅ **Do**
- Use consistent spacing
- Apply smooth transitions
- Provide hover feedback
- Maintain color hierarchy
- Test on multiple devices
- Use semantic HTML
- Add ARIA labels
- Optimize performance

❌ **Don't**
- Mix too many colors
- Use harsh shadows
- Create jarring animations
- Forget accessibility
- Ignore mobile design
- Use inconsistent spacing
- Hardcode values
- Ignore performance

---

## Quick Reference

| Element | Color | Size | Radius | Shadow |
|---------|-------|------|--------|--------|
| Background | #0a0a0a | - | - | - |
| Button | #9333ea | 0.75rem × 1.5rem | 6px | Glow |
| Card | #1a0033 | - | 8px | Subtle |
| Sidebar | #0a0a0a | 60-220px | - | Large |
| Text | #ffffff | 1rem | - | - |
| Border | #9333ea | 1px | - | - |

---

## Implementation Notes

1. **Colors**: Use CSS variables for easy theming
2. **Spacing**: Follow 8px base unit system
3. **Transitions**: Keep under 300ms for responsiveness
4. **Shadows**: Use sparingly for depth
5. **Typography**: Maintain clear hierarchy
6. **Accessibility**: Always include focus states
7. **Performance**: Use GPU-accelerated properties
8. **Testing**: Verify on all target browsers

---

## Design System Version

**Version**: 1.0
**Last Updated**: November 2025
**Status**: Production Ready ✓

For updates or questions, refer to `DESIGN_SPECIFICATION.md`
