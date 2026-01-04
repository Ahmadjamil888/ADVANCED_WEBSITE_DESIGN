# Zehanx Technologies - Customization Guide

## Quick Customizations

### 1. Change Company Contact Information
**File**: `constants/index.ts`

```typescript
export const contactInfo = {
   email: 'your-email@company.com',
   phone: '+92 XXX XXXXXXX',
   company: 'Your Company Name',
   slogan: 'Your Company Slogan'
};
```

### 2. Update Navigation Menu
**File**: `constants/index.ts`

```typescript
export const navigationItems = [
   {
      id: 1,
      title: 'Your Menu Item',
      href: '#your-section'
   },
   // Add more items...
];
```

### 3. Modify Services/Offerings
**File**: `constants/index.ts`

```typescript
export const pricingItems = [
   {
      id: 1,
      title: 'Your Service Name',
      price: null,  // Set to null for no pricing
      btn: "Learn More",
      features: [
         {
            id: 1,
            feature: "Feature 1",
         },
         {
            id: 2,
            feature: "Feature 2",
         },
         // Add more features...
      ]
   },
   // Add more services...
];
```

### 4. Change Company Logo
**Steps**:
1. Replace `public/unnamed.png` with your logo
2. Update favicon reference in `app/layout.tsx`:
   ```typescript
   icons: {
      icon: "/your-logo.png",
   },
   ```
3. Update logo export in `public/index.ts`:
   ```typescript
   export { default as logo } from './your-logo.png';
   ```

### 5. Update Hero Section Text
**File**: `components/Hero.tsx`

```typescript
const phares1 = ["Your headline ", "part 2"];
const phares2 = [
   "Your subheading text here",
];
```

### 6. Modify Services Section
**File**: `components/Pricing.tsx`

```typescript
const phares = ["Your Section Title"];
const phares1 = [
   "Your section description",
   "Line 2",
];
```

### 7. Update Call-to-Action Section
**File**: `components/CallToAction.tsx`

```typescript
const phares1 = ["Your CTA Headline"];
const phares2 = [
   "Your CTA description line 1",
   "Your CTA description line 2",
   "Your CTA description line 3",
];
```

### 8. Customize Footer
**File**: `components/Footer.tsx`

The footer automatically pulls from:
- `contactInfo` from constants
- `footerItems` for navigation links
- `footerSocialsItems` for social media links

Update these in `constants/index.ts`:

```typescript
export const footerItems = [
   {
      id: 1,
      title: 'Your Link',
      href: '#your-section'
   },
   // Add more links...
];

export const footerSocialsItems = [
   {
      id: 1,
      src: instagram,
      href: "https://instagram.com/your-profile"
   },
   // Add more social links...
];
```

### 9. Change Color Scheme
**File**: `components/Hero.tsx`

Update the gradient background:
```typescript
className="w-full h-screen xm:min-h-screen sm:min-h-screen bg-[radial-gradient(ellipse_200%_100%_at_bottom_left,#183EC2,#EAEEFE_80%)]"
```

Change the hex colors:
- `#183EC2` - Primary color
- `#EAEEFE` - Secondary color

### 10. Add New Sections
**Steps**:
1. Create a new component in `components/YourComponent.tsx`
2. Import it in `app/page.tsx`
3. Add it to the JSX in the correct order

Example:
```typescript
// app/page.tsx
import { YourComponent } from "@/components";

export default function App() {
   return (
      <>
         <Hero />
         <YourComponent />  {/* Add your component */}
         <Footer />
      </>
   );
}
```

## Styling

### Tailwind CSS Classes
The project uses Tailwind CSS. Common utilities:

- **Spacing**: `p-4`, `m-2`, `gap-3`
- **Colors**: `text-white`, `bg-black`, `text-[#BCBCBC]`
- **Responsive**: `xm:flex-col`, `sm:flex-col` (mobile), `lg:flex-row` (desktop)
- **Animations**: `hover:text-white`, `transition-all`

### Custom Colors
Update in `tailwind.config.ts` if needed.

## Animations

### Framer Motion
The site uses Framer Motion for smooth animations. Key animations:

- **Text animations**: Fade-in effects on scroll
- **Image animations**: Parallax and scale effects
- **Navigation**: Hide/show on scroll

Modify animation variants in `motion/` folder.

## Adding Social Media Links

**File**: `constants/index.ts`

```typescript
export const footerSocialsItems = [
   {
      id: 1,
      src: instagram,
      href: "https://instagram.com/zehanx"
   },
   {
      id: 2,
      src: linkedin,
      href: "https://linkedin.com/company/zehanx"
   },
   // Add more...
];
```

## Adding Team Members/Testimonials

**File**: `constants/index.ts`

Update the `testimonials` array:

```typescript
export const testimonials = [
   {
      id: 1,
      text: "Testimonial text here",
      src: avatar1,
      name: "Person Name",
      username: "@username",
   },
   // Add more testimonials...
];
```

## Common Issues & Solutions

### Issue: Logo not showing
- Check file path in `public/` folder
- Verify export in `public/index.ts`
- Clear browser cache

### Issue: Colors look different
- Check Tailwind CSS configuration
- Verify hex color codes
- Test in different browsers

### Issue: Animations not smooth
- Check Framer Motion versions
- Verify animation variants
- Test on different devices

### Issue: Mobile layout broken
- Check responsive classes (`xm:`, `sm:`)
- Test on actual mobile device
- Verify Tailwind breakpoints

## Performance Tips

1. **Optimize Images**: Use WebP format when possible
2. **Lazy Load**: Images load on scroll
3. **Code Splitting**: Components load as needed
4. **Caching**: Configure browser caching headers
5. **CDN**: Use CDN for static assets

## SEO Tips

1. Update meta description in `layout.tsx`
2. Add alt text to all images
3. Use semantic HTML
4. Add structured data (JSON-LD)
5. Create sitemap.xml
6. Add robots.txt

## Need Help?

For more information:
- Next.js Docs: https://nextjs.org/docs
- Tailwind CSS: https://tailwindcss.com/docs
- Framer Motion: https://www.framer.com/motion/
- React: https://react.dev

Contact: zehanxtech@gmail.com
