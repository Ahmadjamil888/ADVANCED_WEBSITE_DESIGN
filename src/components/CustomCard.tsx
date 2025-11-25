import React from 'react';

interface CustomCardProps {
  title: string;
  description: string;
  category: string;
  tags?: string[];
  language?: string;
  stars?: number;
  forks?: number;
  url?: string;
  className?: string;
}

export const CustomCard: React.FC<CustomCardProps> = ({
  title,
  description,
  category,
  tags = [],
  language,
  stars,
  forks,
  url,
  className = ""
}) => {
  return (
    <div 
      className={`custom-card ${className}`}
      onClick={() => url && window.open(url, '_blank')}
      style={{
        width: '320px',
        height: '350px',
        padding: '20px',
        color: 'white',
        background: 'linear-gradient(#212121, #212121) padding-box, linear-gradient(145deg, transparent 35%, #2563eb, #3b82f6) border-box',
        border: '2px solid transparent',
        borderRadius: '8px',
        display: 'flex',
        flexDirection: 'column',
        cursor: url ? 'pointer' : 'default',
        transformOrigin: 'right bottom',
        transition: 'all 0.6s cubic-bezier(0.23, 1, 0.320, 1)',
        boxShadow: '0 4px 16px rgba(0,0,0,0.1)'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'rotate(8deg)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'rotate(0deg)';
      }}
    >
      <div className="main-content" style={{ flex: 1 }}>
        <div className="header" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
          <span style={{ fontWeight: '600', color: '#717171', marginRight: '4px' }}>
            {category}
          </span>
          {language && (
            <span style={{ fontWeight: '600', color: '#2563eb', fontSize: '12px' }}>
              {language}
            </span>
          )}
        </div>
        
        <p className="heading" style={{ 
          fontSize: '24px', 
          margin: '24px 0 16px', 
          fontWeight: '600',
          lineHeight: '1.2',
          color: '#ffffff'
        }}>
          {title}
        </p>
        
        <p style={{ 
          color: '#a1a1aa', 
          fontSize: '14px', 
          lineHeight: '1.5',
          marginBottom: '16px',
          minHeight: '60px'
        }}>
          {description || 'No description available'}
        </p>
        
        {tags.length > 0 && (
          <div className="categories" style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px' }}>
            {tags.slice(0, 3).map((tag, index) => (
              <span 
                key={index}
                style={{
                  backgroundColor: '#2563eb',
                  padding: '4px 8px',
                  fontWeight: '600',
                  textTransform: 'uppercase',
                  fontSize: '10px',
                  borderRadius: '50em'
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
      
      {(stars !== undefined || forks !== undefined) && (
        <div className="footer" style={{ 
          fontWeight: '600', 
          color: '#717171', 
          marginTop: '16px',
          paddingTop: '16px',
          borderTop: '1px solid #374151',
          display: 'flex',
          gap: '16px'
        }}>
          {stars !== undefined && (
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              ⭐ {stars}
            </span>
          )}
          {forks !== undefined && (
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              🍴 {forks}
            </span>
          )}
        </div>
      )}
    </div>
  );
};