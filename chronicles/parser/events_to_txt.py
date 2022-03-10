
# %%
for i in range(len(chron)):
    if chron[i]['date'] == 'NaN_before_date':
        continue
    else:
        chrontxt = chron[i]['text']
        filename = str(chron[i]['call_nr']) + '_' + str(chron[i]['date'])[2:-2] + '.txt'
        with open(filename, 'w', encoding='utf-8') as fileout:
            fileout.write('\n'.join(chrontxt))
# %%
