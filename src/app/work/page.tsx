"use client";
import { useEffect, useState } from "react";
import { Column, Heading } from "@/once-ui/components";
import { baseURL } from "@/app/resources";
import { about, person, work } from "@/app/resources/content";
import { Schema } from "@/once-ui/modules";
import { CustomCard } from "@/components/CustomCard";

type Repo = {
  id: number;
  name: string;
  html_url: string;
  description: string;
  language?: string;
  stargazers_count?: number;
  forks_count?: number;
};

export default function Work() {
  const [repos, setRepos] = useState<Repo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("https://api.github.com/users/Ahmadjamil888/repos?per_page=100")
      .then(res => res.json())
      .then(data => {
        // Sort by stars descending
        setRepos(
          data.sort((a: Repo, b: Repo) => (b.stargazers_count || 0) - (a.stargazers_count || 0))
        );
        setLoading(false);
      })
      .catch(error => {
        console.error('Error fetching repos:', error);
        setLoading(false);
      });
  }, []);

  return (
    <Column maxWidth="m">
      <Schema
        as="webPage"
        baseURL={baseURL}
        path={work.path}
        title={work.title}
        description={work.description}
        image={`${baseURL}/og?title=${encodeURIComponent(work.title)}`}
        author={{
          name: person.name,
          url: `${baseURL}${about.path}`,
          image: `${baseURL}${person.avatar}`,
        }}
      />
      <Heading variant="display-strong-s" style={{ marginBottom: 24 }}>
        🚀 zehanxtech Portfolio - AI & Web Development Projects
      </Heading>
      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>Loading our amazing projects...</div>
      ) : (
        <div style={{ 
          display: "flex", 
          flexWrap: "wrap", 
          gap: "24px", 
          justifyContent: "center",
          padding: "20px 0"
        }}>
          {repos.map(repo => (
            <CustomCard
              key={repo.id}
              title={repo.name}
              description={repo.description}
              category="Open Source"
              language={repo.language}
              stars={repo.stargazers_count}
              forks={repo.forks_count}
              url={repo.html_url}
              tags={repo.language ? [repo.language] : []}
            />
          ))}
        </div>
      )}
    </Column>
  );
}