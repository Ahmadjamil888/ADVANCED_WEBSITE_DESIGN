# Dashboard Design Specification

## Color Palette

### Primary Colors
- **Deep Black**: `#0a0a0a` - Main background
- **Dark Gradient**: `#0f0f1a` - Background gradient end
- **Pure Black**: `#000000` - Text and borders

### Accent Colors
- **Deep Purple**: `#9333ea` - Primary accent
- **Purple Gradient Start**: `#1a0033` - Button gradient start
- **Purple Gradient End**: `#330066` - Button gradient end
- **Bright Purple**: `#4d0099` - Hover state

### Neutral Colors
- **White**: `#ffffff` - Primary text
- **Light Gray**: `#ccc` - Secondary text
- **Medium Gray**: `#999` - Tertiary text
- **Dark Gray**: `#333` - Borders

### Semantic Colors
- **Success**: `#00ff00` - Current plan badge
- **Transparent White**: `rgba(255, 255, 255, 0.1)` - Subtle borders
- **Transparent Purple**: `rgba(147, 51, 234, 0.x)` - Accent overlays

---

## Typography

### Font Family
- **Primary**: System fonts (SF Pro Display, Segoe UI, Roboto)
- **Fallback**: `-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`

### Font Sizes
- **Page Title**: 1.8rem (28px) - Bold
- **Section Headers**: 1.5rem (24px) - Bold
- **Card Titles**: 1.2rem (19px) - Semi-bold
- **Body Text**: 1rem (16px) - Regular
- **Small Text**: 0.9rem (14px) - Regular
- **Tiny Text**: 0.75rem (12px) - Regular

### Font Weights
- **Bold**: 700
- **Semi-bold**: 600
- **Regular**: 400

---

## Spacing System

### Base Unit: 0.5rem (8px)

### Common Spacing
- **xs**: 0.25rem (4px)
- **sm**: 0.5rem (8px)
- **md**: 1rem (16px)
- **lg**: 1.5rem (24px)
- **xl**: 2rem (32px)
- **2xl**: 3rem (48px)

### Component Spacing
- **Padding**: 1.5rem - 2rem
- **Margin**: 1rem - 2rem
- **Gap**: 0.5rem - 1.5rem
- **Border Radius**: 4px - 8px

---

## Component Specifications

### Buttons

#### Primary Button (Create AI Model)
```css
background: linear-gradient(135deg, #1a0033 0%, #330066 100%);
border: 1px solid rgba(147, 51, 234, 0.5);
color: #ffffff;
padding: 0.75rem 1.5rem;
border-radius: 6px;
font-weight: 600;
font-size: 1rem;
transition: all 0.3s ease;
```

**Hover State**
```css
background: linear-gradient(135deg, #2d0052 0%, #4d0099 100%);
border-color: rgba(147, 51, 234, 0.8);
box-shadow: 0 0 20px rgba(147, 51, 234, 0.3);
transform: translateY(-2px);
```

#### Secondary Button (Upgrade)
Same as primary button, full width on cards.

### Cards

#### Model Card
```css
background: rgba(20, 10, 40, 0.6);
border: 1px solid rgba(147, 51, 234, 0.3);
border-radius: 8px;
padding: 1.5rem;
transition: all 0.3s ease;
```

**Hover State**
```css
background: rgba(30, 15, 60, 0.8);
border-color: rgba(147, 51, 234, 0.6);
box-shadow: 0 0 15px rgba(147, 51, 234, 0.2);
```

#### Plan Card
```css
background: rgba(15, 8, 35, 0.7);
border: 1px solid rgba(147, 51, 234, 0.4);
border-radius: 8px;
padding: 2rem;
transition: all 0.3s ease;
```

**Hover State**
```css
background: rgba(25, 12, 50, 0.9);
border-color: rgba(147, 51, 234, 0.7);
box-shadow: 0 0 20px rgba(147, 51, 234, 0.25);
transform: translateY(-4px);
```

### Sidebar

#### Sidebar Container
```css
width: 60px (collapsed) / 220px (expanded);
background: linear-gradient(180deg, #0a0a0a 0%, #0f0f1a 100%);
border-right: 1px solid rgba(147, 51, 234, 0.2);
box-shadow: 2px 0 15px rgba(0, 0, 0, 0.5);
transition: width 0.3s ease;
```

#### Sidebar Item
```css
padding: 0.75rem 1rem;
color: #cccccc;
border-left: 3px solid transparent;
transition: all 0.2s ease;
```

**Hover State**
```css
background: rgba(147, 51, 234, 0.1);
color: #ffffff;
border-left-color: rgba(147, 51, 234, 0.5);
```

**Active State**
```css
background: linear-gradient(90deg, rgba(147, 51, 234, 0.3) 0%, transparent 100%);
color: #ffffff;
border-left-color: #9333ea;
box-shadow: inset 0 0 10px rgba(147, 51, 234, 0.2);
```

#### Close Button
```css
width: 32px;
height: 32px;
background: rgba(147, 51, 234, 0.2);
border: 1px solid rgba(147, 51, 234, 0.4);
border-radius: 4px;
color: #cccccc;
transition: all 0.2s ease;
position: absolute;
bottom: 1rem;
right: 1rem;
```

**Hover State**
```css
background: rgba(147, 51, 234, 0.3);
border-color: rgba(147, 51, 234, 0.6);
color: #ffffff;
```

---

## Animations

### Fade In
```css
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
animation: fadeIn 0.5s ease-in;
```

### Hover Lift
```css
transform: translateY(-2px);
transition: transform 0.3s ease;
```

### Glow Effect
```css
box-shadow: 0 0 20px rgba(147, 51, 234, 0.3);
transition: box-shadow 0.3s ease;
```

### Smooth Transition
```css
transition: all 0.3s ease;
```

---

## Responsive Design

### Breakpoints
- **Mobile**: < 640px
- **Tablet**: 640px - 1024px
- **Desktop**: > 1024px

### Grid Layout
```css
grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
gap: 1.5rem;
```

### Sidebar Behavior
- **Mobile**: Hidden by default, toggle with menu button
- **Tablet**: Always visible, collapsible
- **Desktop**: Always visible, expandable on hover

---

## Accessibility

### Color Contrast
- Text on background: Minimum 4.5:1 ratio
- Interactive elements: Minimum 3:1 ratio

### Focus States
```css
outline: 2px solid rgba(147, 51, 234, 0.6);
outline-offset: 2px;
```

### Keyboard Navigation
- Tab through all interactive elements
- Enter/Space to activate buttons
- Escape to close modals/sidebars

### Screen Reader Support
- Semantic HTML elements
- ARIA labels on interactive elements
- Alt text on images

---

## Dark Mode Considerations

### Current Implementation
The dashboard uses a deep dark theme by default:
- Base: `#0a0a0a` (99% black)
- Accent: `#9333ea` (purple)
- Text: `#ffffff` (white)

### Light Mode (Future)
If light mode is added:
- Base: `#ffffff` (white)
- Accent: `#6d28d9` (darker purple)
- Text: `#000000` (black)

---

## Performance Considerations

### CSS Optimization
- Use CSS variables for colors
- Minimize repaints with `will-change`
- Use `transform` for animations (GPU accelerated)
- Avoid `box-shadow` on large elements

### Animation Performance
- Keep animations under 300ms
- Use `ease` timing functions
- Limit simultaneous animations
- Use `requestAnimationFrame` for complex animations

---

## Browser Support

### Supported Browsers
- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: Latest 2 versions
- Mobile browsers: Latest versions

### CSS Features Used
- CSS Grid
- CSS Flexbox
- CSS Gradients
- CSS Transitions
- CSS Transforms
- CSS Variables (optional)

---

## Design System Tokens

### Shadows
```css
--shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
--shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
--shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
--shadow-xl: 0 20px 25px rgba(0, 0, 0, 0.1);
--shadow-glow: 0 0 20px rgba(147, 51, 234, 0.3);
```

### Borders
```css
--border-thin: 1px;
--border-color-light: rgba(255, 255, 255, 0.1);
--border-color-accent: rgba(147, 51, 234, 0.3);
--border-radius-sm: 4px;
--border-radius-md: 6px;
--border-radius-lg: 8px;
```

### Transitions
```css
--transition-fast: 0.2s ease;
--transition-normal: 0.3s ease;
--transition-slow: 0.5s ease;
```

---

## Implementation Checklist

- ✅ Color palette defined
- ✅ Typography system established
- ✅ Spacing system implemented
- ✅ Component specifications documented
- ✅ Animations defined
- ✅ Responsive design planned
- ✅ Accessibility guidelines set
- ✅ Performance optimizations noted
- ✅ Browser support verified
- ✅ Design tokens created

---

## Future Enhancements

1. **CSS Variables**: Implement design tokens as CSS variables
2. **Component Library**: Create reusable component library
3. **Theme Customization**: Allow users to customize colors
4. **Animation Library**: Create animation presets
5. **Responsive Improvements**: Enhanced mobile experience
6. **Accessibility Audit**: Full WCAG 2.1 compliance
7. **Performance Optimization**: Further CSS optimization
8. **Design Documentation**: Storybook integration

---

## Notes

- All colors use hex or rgba format for consistency
- Gradients use 135deg angle for diagonal effect
- Transitions are smooth (0.2s - 0.5s) for better UX
- Purple accent (#9333ea) is used sparingly for emphasis
- Deep black background reduces eye strain
- Hover effects include lift and glow for visual feedback
