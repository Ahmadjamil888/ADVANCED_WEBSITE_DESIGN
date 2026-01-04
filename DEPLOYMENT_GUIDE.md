# Zehanx Technologies Website - Deployment Guide

## Local Development

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn

### Running Locally
```bash
npm install
npm run dev
```
The website will be available at `http://localhost:3000`

## Building for Production

```bash
npm run build
npm start
```

## Project Structure

```
├── app/
│   ├── layout.tsx          # Main layout with metadata
│   └── page.tsx            # Home page
├── components/
│   ├── Header.tsx          # Navigation header
│   ├── Hero.tsx            # Hero section
│   ├── ProductShowcase.tsx # Services showcase
│   ├── Pricing.tsx         # Services cards
│   ├── CallToAction.tsx    # Contact section
│   ├── Footer.tsx          # Footer with contact info
│   └── ...
├── constants/
│   └── index.ts            # Navigation, services, contact info
├── public/
│   ├── unnamed.png         # Company logo (used as favicon)
│   └── ...
└── styles/
    └── globals.css         # Global styles
```

## Key Files to Customize

### 1. Company Information (`constants/index.ts`)
Update the `contactInfo` object:
```typescript
export const contactInfo = {
   email: 'zehanxtech@gmail.com',
   phone: '+92 344 2693910',
   company: 'Zehanx Technologies',
   slogan: 'From concepts to reality'
};
```

### 2. Services (`constants/index.ts`)
Update `pricingItems` array to add/modify services

### 3. Navigation (`constants/index.ts`)
Update `navigationItems` and `footerItems` arrays

### 4. Logo (`public/index.ts`)
The logo is imported from `unnamed.png`. To change:
1. Replace the file or update the import path
2. Update the favicon reference in `app/layout.tsx`

## Deployment Options

### Netlify
1. Connect your GitHub repository
2. Build command: `npm run build`
3. Publish directory: `.next`

### Vercel
1. Import project from GitHub
2. Vercel will auto-detect Next.js
3. Deploy with one click

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Environment Variables
Currently, no environment variables are required. If you add backend services, create a `.env.local` file.

## Performance Optimization

### Already Implemented
- Image optimization with Next.js Image component
- Code splitting and lazy loading
- CSS optimization with Tailwind CSS
- Smooth scrolling with Lenis
- Framer Motion for animations

### Recommended Additions
- Add analytics (Google Analytics, Mixpanel)
- Implement form submission for contact
- Add email notifications
- Set up CDN for static assets

## SEO Optimization

### Already Configured
- Meta title and description
- Favicon
- Responsive design

### Recommended Additions
- Add sitemap.xml
- Add robots.txt
- Add structured data (JSON-LD)
- Add Open Graph tags for social sharing

## Troubleshooting

### Port 3000 Already in Use
```bash
# Kill the process using port 3000
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Mac/Linux
lsof -i :3000
kill -9 <PID>
```

### Build Errors
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

## Contact Information
- **Email**: zehanxtech@gmail.com
- **Phone**: +92 344 2693910

## Support
For issues or questions, contact the development team at zehanxtech@gmail.com
