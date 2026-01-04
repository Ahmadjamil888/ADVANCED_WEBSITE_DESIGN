# New Pages Guide - Zehanx Technologies

## Overview
Three new pages have been created to provide comprehensive information about Zehanx Technologies. All pages maintain consistent design and styling with the main landing page.

## Pages Created

### 1. **About Page** (`/about`)
**Route**: `/about`

#### Sections:
- **Hero Section**: Company headline and tagline
- **Mission & Vision**: Detailed company mission and vision statements
- **Core Values**: Four key values with descriptions
  - Innovation
  - Excellence
  - Collaboration
  - Integrity
- **Why Choose Us**: Three key differentiators
  - Expertise
  - Dedication
  - Innovation
- **CTA Section**: Contact buttons

#### Features:
- Smooth animations on scroll
- Responsive grid layouts
- Professional typography
- Direct contact integration

---

### 2. **Services Page** (`/services`)
**Route**: `/services`

#### Sections:
- **Hero Section**: Services headline and description
- **Services Grid**: 6 comprehensive service cards
  - Artificial Intelligence
  - Machine Learning
  - Data Science
  - Web Development
  - Software Development
  - Consulting
  
  Each service includes:
  - Emoji icon
  - Description
  - 6 key features with checkmarks

- **Our Process**: 6-step development process
  1. Discovery
  2. Planning
  3. Development
  4. Testing
  5. Deployment
  6. Support

- **Technology Stack**: 4 technology categories
  - AI & ML (TensorFlow, PyTorch, Scikit-learn, XGBoost, OpenAI APIs)
  - Web & Backend (React, Next.js, Node.js, Python, PostgreSQL)
  - Data & Analytics (Pandas, NumPy, Tableau, Power BI, Apache Spark)
  - Cloud & DevOps (AWS, Google Cloud, Azure, Docker, Kubernetes)

- **CTA Section**: Contact buttons

#### Features:
- 3-column responsive grid
- Service feature lists with checkmarks
- Staggered animations
- Technology stack display
- Professional card design

---

### 3. **Contact Page** (`/contact`)
**Route**: `/contact`

#### Sections:
- **Hero Section**: Contact headline and description
- **Contact Methods**: 3 main contact options
  - Email (with direct mailto link)
  - Phone (with direct tel link)
  - Company (link to About page)
  
  Each method includes:
  - Icon
  - Title
  - Description
  - Contact value
  - Action button

- **Social Media Links**: 4 social platforms
  - LinkedIn
  - GitHub
  - Twitter
  - Facebook
  
  (Links can be updated in the code)

- **FAQ Section**: 6 frequently asked questions
  - Expandable/collapsible design
  - Smooth animations
  - Professional styling
  
  Questions covered:
  1. Project timeline
  2. Ongoing support
  3. Technology specialization
  4. Quality assurance
  5. Integration with existing systems
  6. Pricing models

- **Quick Contact CTA**: Final contact section

#### Features:
- **No Contact Form**: Direct contact details instead
- Interactive FAQ with expand/collapse
- Social media integration
- Responsive design
- Smooth animations

---

## Design Consistency

All new pages maintain:
- **Color Scheme**: Blue gradients (#183EC2, #001E7F) with white backgrounds
- **Typography**: Consistent heading and paragraph styles
- **Animations**: Framer Motion animations on scroll
- **Components**: Reusable Navbar and Footer
- **Responsive Design**: Mobile, tablet, and desktop layouts
- **Spacing**: Consistent padding and margins
- **Shadows & Borders**: Professional card styling

---

## Navigation Updates

### Updated Navigation Links:
```
- Services → /services
- About → /about
- Portfolio → / (home)
- Contact → /contact
```

### Footer Links:
All footer links have been updated to point to the new pages.

---

## File Structure

```
app/
├── about/
│   └── page.tsx          # About page
├── services/
│   └── page.tsx          # Services page
├── contact/
│   └── page.tsx          # Contact page
├── layout.tsx            # Main layout
└── page.tsx              # Home page

constants/
└── index.ts              # Updated navigation items
```

---

## Customization Guide

### Update Contact Information
**File**: `constants/index.ts`

```typescript
export const contactInfo = {
   email: 'your-email@company.com',
   phone: '+92 XXX XXXXXXX',
   company: 'Your Company Name',
   slogan: 'Your Slogan'
};
```

### Add/Modify Services
**File**: `app/services/page.tsx`

Update the `services` array:
```typescript
const services = [
   {
      id: 1,
      title: 'Your Service',
      icon: '🎯',
      description: 'Description here',
      features: ['Feature 1', 'Feature 2', ...]
   }
];
```

### Update FAQ Questions
**File**: `app/contact/page.tsx`

Update the `faqs` array:
```typescript
const faqs = [
   {
      id: 1,
      question: 'Your question?',
      answer: 'Your answer here'
   }
];
```

### Add Social Media Links
**File**: `app/contact/page.tsx`

Update the `socialLinks` array:
```typescript
const socialLinks = [
   {
      id: 1,
      icon: '💼',
      name: 'LinkedIn',
      description: 'Connect with us',
      link: 'https://linkedin.com/company/your-company'
   }
];
```

### Update Core Values
**File**: `app/about/page.tsx`

Update the `values` array:
```typescript
const values = [
   {
      id: 1,
      title: 'Your Value',
      description: 'Description here'
   }
];
```

---

## SEO Optimization

### Meta Tags to Add
Consider adding meta tags for each page in `layout.tsx` or using Next.js metadata:

```typescript
export const metadata: Metadata = {
   title: 'Page Title | Zehanx Technologies',
   description: 'Page description for SEO',
   keywords: ['keyword1', 'keyword2'],
};
```

### Recommended Additions:
- Open Graph tags for social sharing
- Structured data (JSON-LD)
- Sitemap.xml
- robots.txt

---

## Performance Tips

1. **Images**: All images are optimized with Next.js Image component
2. **Animations**: Framer Motion handles smooth animations efficiently
3. **Code Splitting**: Each page is code-split automatically
4. **Lazy Loading**: Components load on scroll

---

## Testing Checklist

- [ ] Test all pages on desktop
- [ ] Test all pages on tablet
- [ ] Test all pages on mobile
- [ ] Verify all navigation links work
- [ ] Test contact links (mailto, tel)
- [ ] Test FAQ expand/collapse
- [ ] Verify animations are smooth
- [ ] Check responsive layouts
- [ ] Test social media links
- [ ] Verify footer links

---

## Browser Compatibility

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

---

## Deployment Notes

When deploying to production:
1. Update all social media links
2. Verify contact information
3. Test all forms and links
4. Add analytics tracking
5. Set up email notifications for contact inquiries
6. Configure SEO meta tags
7. Test on production environment

---

## Future Enhancements

Consider adding:
1. Blog section
2. Case studies/Portfolio
3. Team member profiles
4. Testimonials carousel
5. Newsletter signup
6. Live chat support
7. Contact form with email notifications
8. Appointment booking system
9. Service pricing comparison
10. Integration with CRM

---

## Support

For questions or issues:
- Email: zehanxtech@gmail.com
- Phone: +92 344 2693910

---

## Version History

- **v1.0** (Nov 25, 2025): Initial release with About, Services, and Contact pages
