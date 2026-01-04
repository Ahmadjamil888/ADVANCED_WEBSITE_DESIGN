# Testing Guide - New Pages

## Quick Links to Test

### Home Page
- **URL**: `http://localhost:3000/`
- **Description**: Main landing page with services overview

### About Page
- **URL**: `http://localhost:3000/about`
- **Description**: Company information, mission, vision, and values

### Services Page
- **URL**: `http://localhost:3000/services`
- **Description**: Detailed service offerings with features and technology stack

### Contact Page
- **URL**: `http://localhost:3000/contact`
- **Description**: Contact information, social links, and FAQ

---

## Testing Checklist

### Navigation Testing
- [ ] Click "About" in header navigation → Should go to `/about`
- [ ] Click "Services" in header navigation → Should go to `/services`
- [ ] Click "Contact" in header navigation → Should go to `/contact`
- [ ] Click "Portfolio" in header navigation → Should go to `/` (home)
- [ ] Click logo → Should go to home page
- [ ] Test mobile menu navigation
- [ ] Test footer navigation links

### About Page Testing
- [ ] Hero section displays correctly
- [ ] Mission and Vision sections are visible
- [ ] Core Values cards display in 2x2 grid
- [ ] "Why Choose Us" section shows 3 cards
- [ ] Contact buttons are clickable
- [ ] Animations play smoothly on scroll
- [ ] Responsive layout on mobile/tablet

### Services Page Testing
- [ ] Hero section displays correctly
- [ ] 6 service cards display in 3-column grid
- [ ] Each service shows icon, title, description, and features
- [ ] Features have checkmarks
- [ ] "Our Process" section shows 6 steps
- [ ] Technology stack displays in 2x2 grid
- [ ] All technology categories are visible
- [ ] Contact buttons work
- [ ] Responsive layout on mobile/tablet

### Contact Page Testing
- [ ] Hero section displays correctly
- [ ] 3 contact method cards display
- [ ] Email button links to mailto
- [ ] Phone button links to tel
- [ ] Company button links to /about
- [ ] Social media section shows 4 platforms
- [ ] Social links open in new tab
- [ ] FAQ section displays all 6 questions
- [ ] FAQ expand/collapse works smoothly
- [ ] Contact CTA buttons work
- [ ] Responsive layout on mobile/tablet

### Link Testing
- [ ] Email links: `mailto:zehanxtech@gmail.com`
- [ ] Phone links: `tel:+92 344 2693910`
- [ ] All navigation links work
- [ ] All footer links work
- [ ] Social media links open in new tab

### Responsive Design Testing

#### Desktop (1920x1080)
- [ ] All content visible
- [ ] Proper spacing and alignment
- [ ] Animations smooth
- [ ] No overflow

#### Tablet (768x1024)
- [ ] Grid layouts adapt to 2 columns
- [ ] Text readable
- [ ] Buttons accessible
- [ ] Navigation works

#### Mobile (375x667)
- [ ] Grid layouts adapt to 1 column
- [ ] Mobile menu works
- [ ] Text readable
- [ ] Buttons accessible
- [ ] No horizontal scroll

### Animation Testing
- [ ] Text animations on scroll
- [ ] Card animations on scroll
- [ ] Smooth transitions
- [ ] No jank or stuttering
- [ ] Animations on mobile are smooth

### Performance Testing
- [ ] Pages load quickly
- [ ] Images load properly
- [ ] No console errors
- [ ] No memory leaks
- [ ] Smooth scrolling

### Browser Testing
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)
- [ ] Mobile Chrome
- [ ] Mobile Safari

### Accessibility Testing
- [ ] Keyboard navigation works
- [ ] Tab order is logical
- [ ] Links are understandable
- [ ] Buttons are clickable
- [ ] Text contrast is good
- [ ] Images have alt text

---

## Common Issues & Solutions

### Issue: Pages not found (404)
**Solution**: 
- Ensure dev server is running: `npm run dev`
- Check file paths in `app/` directory
- Restart dev server

### Issue: Navigation links not working
**Solution**:
- Check `constants/index.ts` for correct paths
- Verify page files exist in correct directories
- Clear browser cache

### Issue: Animations not smooth
**Solution**:
- Check browser performance
- Test in different browser
- Disable browser extensions
- Check for console errors

### Issue: Responsive layout broken
**Solution**:
- Check Tailwind CSS responsive classes
- Test on actual device
- Check browser zoom level
- Clear browser cache

### Issue: Contact links not working
**Solution**:
- Verify email and phone in `constants/index.ts`
- Check browser settings for mailto/tel support
- Test on different browser

---

## Performance Metrics to Check

Using Chrome DevTools:

1. **Lighthouse Score**
   - Performance: > 90
   - Accessibility: > 90
   - Best Practices: > 90
   - SEO: > 90

2. **Core Web Vitals**
   - LCP (Largest Contentful Paint): < 2.5s
   - FID (First Input Delay): < 100ms
   - CLS (Cumulative Layout Shift): < 0.1

3. **Network**
   - Total page size: < 5MB
   - Load time: < 3s

---

## Manual Testing Steps

### Test About Page
1. Navigate to `/about`
2. Scroll through all sections
3. Verify all text is readable
4. Click contact buttons
5. Check responsive layout
6. Test animations

### Test Services Page
1. Navigate to `/services`
2. Scroll through service cards
3. Verify all features are visible
4. Check process steps
5. View technology stack
6. Test responsive layout

### Test Contact Page
1. Navigate to `/contact`
2. Click email button
3. Click phone button
4. Expand FAQ questions
5. Click social media links
6. Test responsive layout

---

## Automated Testing (Optional)

To add automated tests, consider:
- Jest for unit tests
- React Testing Library for component tests
- Cypress for E2E tests
- Playwright for cross-browser testing

---

## Deployment Testing

Before deploying to production:

1. [ ] Run `npm run build` successfully
2. [ ] No build errors or warnings
3. [ ] Test all pages in production build
4. [ ] Verify all links work
5. [ ] Check analytics integration
6. [ ] Test contact notifications
7. [ ] Verify SSL certificate
8. [ ] Test on production domain

---

## Sign-Off Checklist

- [ ] All pages created and accessible
- [ ] Navigation working correctly
- [ ] Responsive design verified
- [ ] Animations smooth
- [ ] Contact links functional
- [ ] No console errors
- [ ] Performance acceptable
- [ ] Accessibility standards met
- [ ] Cross-browser compatibility verified
- [ ] Ready for deployment

---

## Notes

- Dev server URL: `http://localhost:3000`
- Production build: `npm run build && npm start`
- Clear cache if changes not visible: `Ctrl+Shift+Delete` (Chrome)
- Check console for errors: `F12` → Console tab

---

## Support

For issues or questions:
- Email: zehanxtech@gmail.com
- Phone: +92 344 2693910
