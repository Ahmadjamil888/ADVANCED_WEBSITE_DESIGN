import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { remark } from 'remark';
import html from 'remark-html';

const blogsDirectory = path.join(process.cwd(), 'blogs');

export interface BlogPost {
  slug: string;
  title: string;
  date: string;
  excerpt: string;
  author: string;
  readTime: string;
  tags: string[];
  image?: string;
  content: string;
}

export interface BlogMetadata {
  slug: string;
  title: string;
  date: string;
  excerpt: string;
  author: string;
  readTime: string;
  tags: string[];
  image?: string;
}

export function getAllBlogPosts(): BlogMetadata[] {
  try {
    const fileNames = fs.readdirSync(blogsDirectory);
    const allPostsData = fileNames
      .filter((fileName) => fileName.endsWith('.md'))
      .map((fileName) => {
        const slug = fileName.replace(/\.md$/, '');
        const fullPath = path.join(blogsDirectory, fileName);
        const fileContents = fs.readFileSync(fullPath, 'utf8');
        const matterResult = matter(fileContents);

        return {
          slug,
          title: matterResult.data.title || 'Untitled',
          date: matterResult.data.date || new Date().toISOString(),
          excerpt: matterResult.data.excerpt || '',
          author: matterResult.data.author || 'Zehan X Technologies',
          readTime: matterResult.data.readTime || '5 min read',
          tags: matterResult.data.tags || [],
          image: matterResult.data.image || '/blog/default-blog-image.jpg',
        };
      });

    return allPostsData.sort((a, b) => (a.date < b.date ? 1 : -1));
  } catch (error) {
    console.error('Error reading blog posts:', error);
    return [];
  }
}

export async function getBlogPost(slug: string): Promise<BlogPost | null> {
  try {
    const fullPath = path.join(blogsDirectory, `${slug}.md`);
    const fileContents = fs.readFileSync(fullPath, 'utf8');
    const matterResult = matter(fileContents);

    const processedContent = await remark()
      .use(html)
      .process(matterResult.content);
    const contentHtml = processedContent.toString();

    return {
      slug,
      title: matterResult.data.title || 'Untitled',
      date: matterResult.data.date || new Date().toISOString(),
      excerpt: matterResult.data.excerpt || '',
      author: matterResult.data.author || 'Zehan X Technologies',
      readTime: matterResult.data.readTime || '5 min read',
      tags: matterResult.data.tags || [],
      image: matterResult.data.image || '/blog/default-blog-image.jpg',
      content: contentHtml,
    };
  } catch (error) {
    console.error(`Error reading blog post ${slug}:`, error);
    return null;
  }
}

export function getBlogSlugs(): string[] {
  try {
    const fileNames = fs.readdirSync(blogsDirectory);
    return fileNames
      .filter((fileName) => fileName.endsWith('.md'))
      .map((fileName) => fileName.replace(/\.md$/, ''));
  } catch (error) {
    console.error('Error reading blog slugs:', error);
    return [];
  }
}