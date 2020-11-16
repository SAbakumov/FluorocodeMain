
genomes = {'Other','CP000948','CP014051','CP014787','CP039296'};
classes = 5;
Total = zeros(1,classes );
AllTotal = zeros(1,classes );
for i = 1:length(y)
    for j=1:classes 
%        if y(i,j)==1
%            AllTotal(j) = AllTotal(j)+1;
%            
%        end
       if y(i,j)>=0.99
           Total(j) = Total(j)+1;
       end
        
    end
    
end
bar(1:5,Total./2400)
xticklabels(genomes)
set(gca,'FontSize',24)
xtickangle(45)